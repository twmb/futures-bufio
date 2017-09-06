use futures::{Async, Future, Poll};
use futures_cpupool::CpuPool;

use std::io::{self, Read};
use std::mem;
use std::cell::RefCell;
use std::rc::Rc;

/// Adds a memory buffer to R. We separate this from `Inner` so that we can send it across threads.
struct Membuf<R> {
    r: R,
    buf: Box<[u8]>,
    pos: usize,
    cap: usize,
}

/// Contains a memory-buffer wrapped R and the `CpuPool` that runs the disk operations. We take out
/// of `mb` to send the membuf to a thread for writing. The `CpuFuture` always returns membuf; we
/// put it back when the future is done.
struct Inner<R> {
    mb: Option<Membuf<R>>,
    pool: CpuPool,
}

struct ReadFuture<R> {
    state: ReadState<R>,
}

// These types are quite verbose, so we alias them.

type BoxFuture<I, E> = Box<Future<Item = I, Error = E>>;
type RcInner<R> = Rc<RefCell<Option<Inner<R>>>>;
type OkWork<R> = (Inner<R>, Box<[u8]>, usize);
type ErrWork<R> = (Inner<R>, Box<[u8]>, io::Error);

/// Wraps `R` with the original buffer being read into and the number of bytes read.
type OkRead = (Box<[u8]>, usize);
/// Wraps `R` with the original buffer being read into and the error encountered while reading.
type ErrRead = (Box<[u8]>, io::Error);

enum ReadState<R> {
    Unstarted((RcInner<R>, Box<[u8]>)),
    Working((RcInner<R>, BoxFuture<OkWork<R>, ErrWork<R>>)),
    Done,
}

impl<R> Future for ReadFuture<R>
where
    R: Read + Send + 'static,
{
    type Item = OkRead;
    type Error = ErrRead;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        loop {
            match mem::replace(&mut self.state, ReadState::Done) {
                ReadState::Unstarted((rc, mut buf)) => {
                    // Try to take ownership of inner. If we cannot, something else is working. We need
                    // full ownership here so that we do not try stuffing references in a Box.
                    let take = rc.borrow_mut().take();
                    let mut inner = match take {
                        Some(inner) => inner,
                        None => {
                            self.state = ReadState::Unstarted((rc, buf));
                            return Ok(Async::NotReady);
                        }
                    };

                    const U8READ: &str = "&[u8] reads never error";

                    // Try to read from our memory buffer without taking it out of the option.
                    let mut rem = buf.len();
                    let mut at = 0;

                    {
                        let mb = inner.mb.as_mut().take().expect("membuf should exist here");
                        if mb.pos != mb.cap {
                            at = (&mb.buf[mb.pos..mb.cap]).read(&mut buf).expect(U8READ);
                            rem -= at;
                            mb.pos += at;
                        }
                    }

                    if rem == 0 {
                        *rc.borrow_mut() = Some(inner);
                        return Ok(Async::Ready((buf, at)));
                    }

                    // Our memory buffer is empty and we still need to fill buf. Rip our memory buffer
                    // out so we can send it to the other thread - we may need to fill it.
                    let mut mb = inner.mb.take().expect("membuf should exist here");

                    // Spin our reads into the inner's pool.
                    let fut = inner.pool.spawn_fn(move || {
                        let block = if mb.cap > 0 { rem - rem % mb.cap } else { rem };
                        if block > 0 {
                            let (block_read, err) =
                                try_read_full(&mut mb.r, &mut buf[at..at + block]);
                            if let Some(e) = err {
                                return Err((mb, buf, e));
                            }

                            at += block_read;
                            rem -= block_read;
                            if rem == 0 {
                                return Ok((mb, buf, at));
                            }
                        }

                        let (buf_read, err) = try_read_full(&mut mb.r, &mut mb.buf);
                        match err {
                            Some(e) => Err((mb, buf, e)),
                            None => {
                                mb.cap = buf_read;
                                mb.pos = (&mb.buf[..mb.cap]).read(&mut buf[at..]).expect(U8READ);
                                at += mb.pos;
                                Ok((mb, buf, at))
                            }
                        }
                    });

                    // Update our state to working, saving the rc so that we can re-fill the option.
                    self.state = ReadState::Working((
                        rc,
                        Box::new(fut.then(move |res| match res {
                            Ok((mb, buf, n)) => {
                                inner.mb = Some(mb);
                                Ok((inner, buf, n))
                            }
                            Err((mb, buf, e)) => {
                                inner.mb = Some(mb);
                                Err((inner, buf, e))
                            }
                        })),
                    ));
                }

                ReadState::Working((rc, mut f)) => {
                    match f.poll() {
                        Ok(Async::NotReady) => {
                            self.state = ReadState::Working((rc, f));
                            return Ok(Async::NotReady);
                        }
                        Ok(Async::Ready((inner, buf, n))) => {
                            *rc.borrow_mut() = Some(inner);
                            return Ok(Async::Ready((buf, n)));
                        }
                        Err((inner, buf, e)) => {
                            *rc.borrow_mut() = Some(inner);
                            return Err((buf, e));
                        }
                    }
                }

                ReadState::Done => panic!("cannot poll done ReadFuture twice"),
            }
        }
    }
}

/// Adds buffering to any reader, similar to the [standard `BufReader`], but performs non-buffer
/// reads in a thread pool.
///
/// [standard `BufReader`]: https://doc.rust-lang.org/std/io/struct.BufReader.html
///
/// All reads are returned as futures.
///
/// This reader is most useful for wrapping readers that never block or cannot return EWOULDBLOCK,
/// but are slow. Notably, this is useful for wrapping `io::File`.
///
/// All reads must take and own the `BufReader` and the buffer being written for the duration of
/// the read.
///
/// # Examples
/// ```
/// # extern crate futures_cpupool;
/// # extern crate futures;
/// # extern crate futures_bufio;
/// #
/// # use futures::Future;
/// # use futures_cpupool::CpuPool;
/// # use futures_bufio::BufReader;
/// # use std::io;
/// # fn main() {
/// let f = io::Cursor::new(b"normally, we would open a file here here".to_vec());
/// let pool = CpuPool::new(1);
/// let reader = BufReader::with_pool_and_capacity(pool, 10, f);
///
/// let buf = vec![0; 10].into_boxed_slice();
/// let (buf, n) = reader.try_read_full(buf).wait().unwrap_or_else(|(_, e)| {
///     // in real usage, we have the option to deconstruct our BufReader or reuse buf here
///     panic!("unable to read full: {}", e);
/// });
/// assert_eq!(n, 10);
/// assert_eq!(&buf[..n], b"normally, ");
/// # }
/// ```
pub struct BufReader<R> {
    inner: RcInner<R>,
}

impl<R: Read + Send + 'static> BufReader<R> {
    /// Creates and returns a new `BufReader` with an internal buffer of size `cap`.
    ///
    /// # Examples
    /// ```ignore
    /// let f = fs::File::open("fubar.txt")?;
    /// let pool = CpuPool::new(1);
    /// let reader = BufReader::with_pool_and_capacity(pool, 4<<10, f);
    /// ```
    pub fn with_pool_and_capacity(pool: CpuPool, cap: usize, inner: R) -> BufReader<R> {
        BufReader::with_pool_and_buf(pool, vec![0; cap].into_boxed_slice(), inner)
    }

    /// Creates and returns a new `BufReader` with `buf` as the internal buffer.
    ///
    /// # Examples
    /// ```ignore
    /// let f = fs::File::open("fubar.txt")?;
    /// let pool = CpuPool::new(1);
    /// let buf = vec![0; 4096].into_boxed_slice();
    /// let reader = BufReader::with_pool_and_buf(pool, buf, f);
    /// ```
    pub fn with_pool_and_buf(pool: CpuPool, buf: Box<[u8]>, inner: R) -> BufReader<R> {
        let cap = buf.len();
        BufReader {
            inner: Rc::new(RefCell::new(Some(Inner {
                mb: Some(Membuf {
                    r: inner,
                    buf: buf,
                    pos: cap,
                    cap: cap,
                }),
                pool: pool,
            }))),
        }
    }

    // Gets a reference to the underlying reader.
    //
    // It is likely invalid to read directly from the underlying reader and then use the `BufReader`
    // again.
    // pub fn get_ref(&self) -> &R {
    // &self.inner
    // }
    //
    // Gets a mutable reference to the underlying reader.
    //
    // It is likely invalid to read directly from the underlying reader and then use the `BufReader`
    // again.
    // pub fn get_mut(&mut self) -> &R {
    // &mut self.inner
    // }
    //
    // Sets the `BufReader`s internal buffer position to `pos`.
    //
    //
    // This is _highly_ unsafe for the following reasons:
    //
    //   - the internal buffer may have uninitialized memory, and moving the read position back
    //     will mean reading uninitialized memory
    //
    //   - the pos is not validated, meaning it is possible to move the pos past the end of the
    //     internal buffer. This will cause a panic on the next use of `try_read_full`.
    //
    //   - it is possible to move _past_ the internal "capacity" end position, meaning a read may
    //     panic due to the beginning of a read being after its end.
    //
    // This function should only be used for setting up a new `BufReader` with a buffer that
    // contains known, existing contents.
    //
    // # Examples
    // ```
    // # extern crate futures_cpupool;
    // # extern crate futures;
    // # extern crate futures_bufio;
    // #
    // # use futures::Future;
    // # use futures_cpupool::CpuPool;
    // # use futures_bufio::BufReader;
    // # use std::io;
    // # fn main() {
    // let f = io::Cursor::new(vec![]);
    // let pool = CpuPool::new(1);
    // let mut buf = vec![0; 4096].into_boxed_slice();
    //
    // let p = b"pre-existing text";
    // let buf_len = buf.len();
    //
    // // copy some known text to our buffer - note it must be at the end
    // &mut buf[buf_len-p.len()..].copy_from_slice(p);
    //
    // let mut reader = BufReader::with_pool_and_buf(pool, buf, f);
    //
    // // unsafely move the reader's position to the beginning of our known text, and read it
    // unsafe { reader.set_pos(buf_len-p.len()); }
    // let (_, b, _) = reader
    //     .try_read_full(vec![0; p.len()].into_boxed_slice())
    //     .wait()
    //     .unwrap_or_else(|(_, _, e)| {
    //         panic!("unable to read: {}", e);
    //     });
    //
    // // our read should be all of our known contents
    // assert_eq!(&*b, p);
    // # }
    // ```
    // pub unsafe fn set_pos(&mut self, pos: usize) {
    // self.pos = pos;
    // }
    //
    // Returns the internal components of a `BufReader`, allowing reuse. This
    // is unsafe because it does not zero the memory of the buffer, meaning
    // the buffer could countain uninitialized memory.
    //
    // # Examples
    // ```
    // # extern crate futures_cpupool;
    // # extern crate futures_bufio;
    // #
    // # use futures_cpupool::CpuPool;
    // # use futures_bufio::BufReader;
    // # use std::io;
    // # fn main() {
    // let f = io::Cursor::new(b"foo text".to_vec());
    // let pool = CpuPool::new(1);
    // let reader = BufReader::with_pool_and_capacity(pool, 4<<10, f);
    //
    // let (f, buf, pool) = unsafe { reader.components() };
    // assert_eq!(f.get_ref(), b"foo text");
    // assert_eq!(buf.len(), 4<<10);
    // # }
    // ```
    // pub unsafe fn components(mut self) -> (R, Box<[u8]>, CpuPool) {
    // let r = mem::replace(&mut self.inner, mem::uninitialized());
    // let buf = mem::replace(&mut self.buf, mem::uninitialized());
    // let mut pool = mem::replace(&mut self.pool, mem::uninitialized());
    // let pool = pool.take().expect(EXP_POOL);
    // mem::forget(self);
    // (r, buf, pool)
    // }
    //

    // Reads into `buf` until `buf` is filled or the underlying reader returns a zero read (hits
    // EOF).
    //
    // This returns the buffer and the number of bytes read. The buffer may need sized down on use
    // if this returns with a short read.
    //
    // If used on `io::File`'s, `BufReader` could be valuable for performing page-aligned reads.
    // In this case, once this function returns a short read, we reached EOF and any futures
    // reads may be un-aligned.
    //
    // # Examples
    // ```
    // # extern crate futures_cpupool;
    // # extern crate futures;
    // # extern crate futures_bufio;
    // #
    // # use futures::Future;
    // # use futures_cpupool::CpuPool;
    // # use futures_bufio::BufReader;
    // # use std::io;
    // # fn main() {
    // let f = io::Cursor::new(b"foo text".to_vec());
    // let pool = CpuPool::new(1);
    // let reader = BufReader::with_pool_and_capacity(pool, 10, f);
    //
    // let buf = vec![0; 10].into_boxed_slice();
    // let (reader, buf, n) = reader.try_read_full(buf).wait().unwrap_or_else(|(_, _, e)| {
    //     // in real usage, we have the option to deconstruct our BufReader or reuse buf here
    //     panic!("unable to read full: {}", e);
    // });
    // assert_eq!(n, 8);
    // assert_eq!(&*buf, b"foo text\0\0");
    // assert_eq!(&buf[..n], b"foo text");
    // # }
    // ```
    pub fn try_read_full(&self, buf: Box<[u8]>) -> impl Future<Item = OkRead, Error = ErrRead> {
        ReadFuture { state: ReadState::Unstarted((self.inner.clone(), buf)) }
    }
}

fn try_read_full<R: Read>(r: &mut R, mut buf: &mut [u8]) -> (usize, Option<io::Error>) {
    let mut nn: usize = 0;
    while !buf.is_empty() {
        match r.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
                nn += n;
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return (nn, Some(e)),
        }
    }
    (nn, None)
}

#[test]
fn test_read() {
    use std::fs;
    use std::io::Write;

    // create the test file
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open("bar.txt")
        .expect("unable to exclusively create foo")
        .write_all(
            "Strapped down to my bed, feet cold, eyes red. I'm out of my head. Am I alive? Am I \
             dead?"
                .as_bytes(),
        )
        .expect("unable to write all");


    // create our buffered reader
    let f = BufReader::with_pool_and_capacity(
        CpuPool::new(1),
        10,
        fs::OpenOptions::new().read(true).open("bar.txt").expect(
            "foo does not exist?",
        ),
    );

    // disk read, no blocks
    let (buf, n) = f.try_read_full(vec![0; 5].into()).wait().unwrap_or_else(
        |(_, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(&buf[..n], b"Strap");
}
// mem read only
// let (f, buf, n) = f.try_read_full(vec![0; 2].into()).wait().unwrap_or_else(
// |(_, _, e)| {
// panic!("unable to read: {}", e)
// },
// );
// assert_eq!(n, 2);
// assert_eq!(&*buf, b"pe");
// assert_eq!(f.pos, 7);
// assert_eq!(f.cap, 10);
// assert_eq!(&*f.buf, b"Strapped d");
//
// mem (3) + disk blocks (20) + more mem (2)
// let (f, buf, n) = f.try_read_full(vec![0; 25].into()).wait().unwrap_or_else(
// |(_, _, e)| {
// panic!("unable to read: {}", e)
// },
// );
// assert_eq!(n, 25);
// assert_eq!(&*buf, b"d down to my bed, feet co");
// assert_eq!(f.pos, 2);
// assert_eq!(f.cap, 10);
// assert_eq!(&*f.buf, b"cold, eyes");
//
// mem (8) + disk block (10)
// let (f, buf, n) = f.try_read_full(vec![0; 18].into()).wait().unwrap_or_else(
// |(_, _, e)| {
// panic!("unable to read: {}", e)
// },
// );
// assert_eq!(n, 18);
// assert_eq!(&*buf, b"ld, eyes red. I'm ");
// assert_eq!(f.pos, 10);
// assert_eq!(f.cap, 10);
// assert_eq!(&*f.buf, b"cold, eyes"); // non-reset buf
//
// disk block (10)
// let (f, buf, n) = f.try_read_full(vec![0; 10].into()).wait().unwrap_or_else(
// |(_, _, e)| {
// panic!("unable to read: {}", e)
// },
// );
// assert_eq!(n, 10);
// assert_eq!(&*buf, b"out of my ");
// assert_eq!(f.pos, 10);
// assert_eq!(f.cap, 10);
// assert_eq!(&*f.buf, b"cold, eyes");
//
// disk block (20) + mem (9) (over-read by one byte)
// let (f, buf, n) = f.try_read_full(vec![0; 29].into()).wait().unwrap_or_else(
// |(_, _, e)| {
// panic!("unable to read: {}", e)
// },
// );
// assert_eq!(n, 28);
// assert_eq!(&*buf, b"head. Am I alive? Am I dead?\0");
// assert_eq!(f.pos, 8);
// assert_eq!(f.cap, 8);
// assert_eq!(&*f.buf, b" I dead?es");
//
// let (f, buf, n) = f.try_read_full(vec![0; 2].into()).wait().unwrap_or_else(
// |(_, _, e)| {
// panic!("unable to read: {}", e)
// },
// );
// assert_eq!(n, 0);
// assert_eq!(&*buf, b"\0\0");
// assert_eq!(f.pos, 0);
// assert_eq!(f.cap, 0);
// assert_eq!(&*f.buf, b" I dead?es");
//
// fs::remove_file("bar.txt").expect("expected file to be removed");
// }
//

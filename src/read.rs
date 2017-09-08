use futures::Future;
use futures::future::{Either, ok};
use futures_cpupool::CpuPool;

use std::ops::DerefMut;

use std::io::{self, Read};
use common::U8READ;

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
/// let buf = vec![0; 10];
/// let (reader, buf, n) = reader.try_read_full(buf).wait().unwrap_or_else(|(_, _, e)| {
///     // in real usage, we have the option to deconstruct our BufReader or reuse buf here
///     panic!("unable to read full: {}", e);
/// });
/// assert_eq!(n, 10);
/// assert_eq!(&buf[..n], b"normally, ");
/// # }
/// ```
pub struct BufReader<R> {
    reader: BufReaderInner<R>,
    pool: CpuPool,
}

struct BufReaderInner<R> {
    inner: R,
    buf: Box<[u8]>,
    pos: usize,
    cap: usize,
}

/// Wraps `R` with the original buffer being read into and the number of bytes read.
type OkRead<R, B> = (R, B, usize);
/// Wraps `R` with the original buffer being read into and the error encountered while reading.
type ErrRead<R, B> = (R, B, io::Error);

impl<R: Read + Send + 'static> BufReader<R> {
    /// Creates and returns a new `BufReader` with an internal buffer of size `cap`.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufReader;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(b"foo text".to_vec());
    /// let pool = CpuPool::new(1);
    /// let reader = BufReader::with_pool_and_capacity(pool, 4<<10, f);
    /// # }
    /// ```
    pub fn with_pool_and_capacity(pool: CpuPool, cap: usize, inner: R) -> BufReader<R> {
        let mut buf = Vec::with_capacity(cap);
        unsafe {
            buf.set_len(cap);
        }
        BufReader::with_pool_and_buf(pool, buf.into_boxed_slice(), inner)
    }

    /// Creates and returns a new `BufReader` with `buf` as the internal buffer.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufReader;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(b"foo text".to_vec());
    /// let pool = CpuPool::new(1);
    /// let buf = vec![0; 4096].into_boxed_slice();
    /// let reader = BufReader::with_pool_and_buf(pool, buf, f);
    /// # }
    /// ```
    pub fn with_pool_and_buf(pool: CpuPool, buf: Box<[u8]>, inner: R) -> BufReader<R> {
        let cap = buf.len();
        BufReader {
            reader: BufReaderInner {
                inner,
                buf: buf,
                pos: cap,
                cap: cap,
            },
            pool: pool,
        }
    }

    /// Gets a reference to the underlying reader.
    ///
    /// It is likely invalid to read directly from the underlying reader and then use the `BufReader`
    /// again.
    pub fn get_ref(&self) -> &R {
        &self.reader.inner
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// It is likely invalid to read directly from the underlying reader and then use the `BufReader`
    /// again.
    pub fn get_mut(&mut self) -> &R {
        &mut self.reader.inner
    }

    /// Sets the `BufReader`s internal buffer position to `pos`.
    ///
    ///
    /// This is _highly_ unsafe for the following reasons:
    ///
    ///   - the internal buffer may have uninitialized memory, and moving the read position back
    ///     will mean reading uninitialized memory
    ///
    ///   - the pos is not validated, meaning it is possible to move the pos past the end of the
    ///     internal buffer. This will cause a panic on the next use of `try_read_full`.
    ///
    ///   - it is possible to move _past_ the internal "capacity" end position, meaning a read may
    ///     panic due to the beginning of a read being after its end.
    ///
    /// This function should only be used for setting up a new `BufReader` with a buffer that
    /// contains known, existing contents.
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
    /// let f = io::Cursor::new(vec![]);
    /// let pool = CpuPool::new(1);
    /// let mut buf = vec![0; 4096].into_boxed_slice();
    ///
    /// let p = b"pre-existing text";
    /// let buf_len = buf.len();
    ///
    /// // copy some known text to our buffer - note it must be at the end
    /// &mut buf[buf_len-p.len()..].copy_from_slice(p);
    ///
    /// let mut reader = BufReader::with_pool_and_buf(pool, buf, f);
    ///
    /// // unsafely move the reader's position to the beginning of our known text, and read it
    /// unsafe { reader.set_pos(buf_len-p.len()); }
    /// let (_, b, _) = reader
    ///     .try_read_full(vec![0; p.len()])
    ///     .wait()
    ///     .unwrap_or_else(|(_, _, e)| {
    ///         panic!("unable to read: {}", e);
    ///     });
    ///
    /// // our read should be all of our known contents
    /// assert_eq!(&*b, p);
    /// # }
    /// ```
    pub unsafe fn set_pos(&mut self, pos: usize) {
        self.reader.pos = pos;
    }

    /// Returns the internal components of a `BufReader`, allowing reuse. This
    /// is unsafe because it does not zero the memory of the buffer, meaning
    /// the buffer could countain uninitialized memory.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufReader;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(b"foo text".to_vec());
    /// let pool = CpuPool::new(1);
    /// let reader = BufReader::with_pool_and_capacity(pool, 4<<10, f);
    ///
    /// let (f, buf, pool) = unsafe { reader.components() };
    /// assert_eq!(f.get_ref(), b"foo text");
    /// assert_eq!(buf.len(), 4<<10);
    /// # }
    /// ```
    pub unsafe fn components(self) -> (R, Box<[u8]>, CpuPool) {
        let BufReader { reader: BufReaderInner { inner: r, buf, .. }, pool, .. } = self;
        (r, buf, pool)
    }

    /// Reads into `buf` until `buf` is filled or the underlying reader returns a zero read (hits
    /// EOF).
    ///
    /// This returns the buffer and the number of bytes read. The buffer may need sized down on use
    /// if this returns with a short read.
    ///
    /// If used on `io::File`'s, `BufReader` could be valuable for performing page-aligned reads.
    /// In this case, once this function returns a short read, we reached EOF and any futures
    /// reads may be un-aligned.
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
    /// let f = io::Cursor::new(b"foo text".to_vec());
    /// let pool = CpuPool::new(1);
    /// let reader = BufReader::with_pool_and_capacity(pool, 10, f);
    ///
    /// let buf = vec![0; 10];
    /// let (reader, buf, n) = reader.try_read_full(buf).wait().unwrap_or_else(|(_, _, e)| {
    ///     // in real usage, we have the option to deconstruct our BufReader or reuse buf here
    ///     panic!("unable to read full: {}", e);
    /// });
    /// assert_eq!(n, 8);
    /// assert_eq!(&*buf, b"foo text\0\0");
    /// assert_eq!(&buf[..n], b"foo text");
    /// # }
    /// ```
    pub fn try_read_full<B>(
        mut self,
        mut buf: B,
    ) -> impl Future<Item = OkRead<Self, B>, Error = ErrRead<Self, B>>
    where
        B: DerefMut<Target = [u8]> + Send + 'static,
    {
        let mut rem = buf.len();
        let mut at = 0;

        if self.reader.pos != self.reader.cap {
            at = (&self.reader.buf[self.reader.pos..self.reader.cap]).read(&mut buf).expect(
                U8READ,
            );
            rem -= at;
            self.reader.pos += at;

            if rem == 0 {
                return Either::A(ok::<OkRead<Self, B>, ErrRead<Self, B>>((self, buf, at)));
            }
        }
        // self.pos == self.cap

        let block = if self.reader.cap > 0 {
            rem - rem % self.reader.cap
        } else {
            rem
        };

        let BufReader { mut reader, pool } = self;

        let fut = pool.spawn_fn(move || {
            if block > 0 {
                let (block_read, err) = try_read_full(&mut reader.inner, &mut buf[at..at + block]);
                if let Some(e) = err {
                    return Err((reader, buf, e));
                }

                at += block_read;
                rem -= block_read;
                if rem == 0 {
                    return Ok((reader, buf, at));
                }
            }

            let (buf_read, err) = try_read_full(&mut reader.inner, &mut reader.buf);
            match err {
                Some(e) => Err((reader, buf, e)),
                None => {
                    reader.cap = buf_read;
                    reader.pos = (&reader.buf[..reader.cap]).read(&mut buf[at..]).expect(U8READ);
                    at += reader.pos;
                    Ok((reader, buf, at))
                }
            }
        });

        Either::B(fut.then(|res| match res {
            Ok((reader, buf, at)) => {
                Ok((BufReader { reader, pool }, buf, at))
            }
            Err((reader, buf, at)) => {
                Err((BufReader { reader, pool }, buf, at))
            }
        }))
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
    assert_eq!(f.reader.pos, 10);

    // disk read, no blocks
    let (f, buf, n) = f.try_read_full(vec![0; 5]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 5);
    assert_eq!(&*buf, b"Strap");
    assert_eq!(f.reader.pos, 5);
    assert_eq!(f.reader.cap, 10);
    assert_eq!(&*f.reader.buf, b"Strapped d");

    // mem read only
    let (f, buf, n) = f.try_read_full(vec![0; 2]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 2);
    assert_eq!(&*buf, b"pe");
    assert_eq!(f.reader.pos, 7);
    assert_eq!(f.reader.cap, 10);
    assert_eq!(&*f.reader.buf, b"Strapped d");

    // mem (3) + disk blocks (20) + more mem (2)
    let (f, buf, n) = f.try_read_full(vec![0; 25]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 25);
    assert_eq!(&*buf, b"d down to my bed, feet co");
    assert_eq!(f.reader.pos, 2);
    assert_eq!(f.reader.cap, 10);
    assert_eq!(&*f.reader.buf, b"cold, eyes");

    // mem (8) + disk block (10)
    let (f, buf, n) = f.try_read_full(vec![0; 18]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 18);
    assert_eq!(&*buf, b"ld, eyes red. I'm ");
    assert_eq!(f.reader.pos, 10);
    assert_eq!(f.reader.cap, 10);
    assert_eq!(&*f.reader.buf, b"cold, eyes"); // non-reset buf

    // disk block (10)
    let (f, buf, n) = f.try_read_full(vec![0; 10]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 10);
    assert_eq!(&*buf, b"out of my ");
    assert_eq!(f.reader.pos, 10);
    assert_eq!(f.reader.cap, 10);
    assert_eq!(&*f.reader.buf, b"cold, eyes");

    // disk block (20) + mem (9) (over-read by one byte)
    let (f, buf, n) = f.try_read_full(vec![0; 29]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 28);
    assert_eq!(&*buf, b"head. Am I alive? Am I dead?\0");
    assert_eq!(f.reader.pos, 8);
    assert_eq!(f.reader.cap, 8);
    assert_eq!(&*f.reader.buf, b" I dead?es");

    let (f, buf, n) = f.try_read_full(vec![0; 2]).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to read: {}", e)
        },
    );
    assert_eq!(n, 0);
    assert_eq!(&*buf, b"\0\0");
    assert_eq!(f.reader.pos, 0);
    assert_eq!(f.reader.cap, 0);
    assert_eq!(&*f.reader.buf, b" I dead?es");

    fs::remove_file("bar.txt").expect("expected file to be removed");
}

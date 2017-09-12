use futures::Future;
use futures::future::{Either, ok};
use futures_cpupool::CpuPool;

use std::io::{self, Write};
use std::ops::Deref;

use common::*;

/// Adds buffering to any writer, similar to the [standard `BufWriter`], but performs non-buffer
/// writes in a thread pool.
///
/// [standard `BufWriter`]: https://doc.rust-lang.org/std/io/struct.BufWriter.html
///
/// All writes are returned as futures.
///
/// This writer is most useful for wrapping writers that never block or cannot return EWOULDBLOCK,
/// but are slow. Notably, this is useful for wrapping `io::File`.
///
/// All writes must take and own the `BufWriter` and the buffer being written for the duration of
/// the write.
///
/// Note that unlike the standard `BufWriter`, this `BufWriter` is _not_ automatically flushed on
/// drop. Users must call [`flush_buf`] and potentially [`flush_inner`] to flush contents before
/// dropping.
///
/// [`flush_buf`]: struct.BufWriter.html#method.flush_buf
/// [`flush_inner`]: struct.BufWriter.html#method.flush_inner
///
/// # Examples
/// ```
/// # extern crate futures_cpupool;
/// # extern crate futures;
/// # extern crate futures_bufio;
/// #
/// # use futures::Future;
/// # use futures_cpupool::CpuPool;
/// # use futures_bufio::BufWriter;
/// # use std::io;
/// # fn main() {
/// let f = io::Cursor::new(vec![]);
/// let pool = CpuPool::new(1);
/// let writer = BufWriter::with_pool_and_capacity(pool, 4096, f);
///
/// let buf = b"many small writes".to_vec();
/// let (writer, buf) = writer.write_all(buf).wait().unwrap_or_else(|(_, _, e)| {
///     // in real usage, we have the option to deconstruct our BufWriter or reuse buf here
///     panic!("unable to read full: {}", e);
/// });
/// assert_eq!(&*buf, b"many small writes"); // we can reuse buf
/// # }
/// ```
pub struct BufWriter<W> {
    writer: BufWriterInner<W>,
    pool: CpuPool,
}

struct BufWriterInner<W> {
    inner: W,
    buf: Box<[u8]>,
    pos: usize,
    w_start: usize,
}

/// Wraps `W` with the original buffer being written.
type OkWrite<W, B> = (W, B);
/// Wraps `W` with the original buffer being written and the error encountered while writing.
type ErrWrite<W, B> = (W, B, io::Error);

impl<W: Write + Send + 'static> BufWriter<W> {
    /// Creates and returns a new `BufWriter` with an internal buffer of size `cap`.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufWriter;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(vec![]);
    /// let pool = CpuPool::new(1);
    /// let writer = BufWriter::with_pool_and_capacity(pool, 4<<10, f);
    /// # }
    /// ```
    pub fn with_pool_and_capacity(pool: CpuPool, cap: usize, inner: W) -> BufWriter<W> {
        let mut buf = Vec::with_capacity(cap);
        unsafe {
            buf.set_len(cap);
        }
        BufWriter::with_pool_and_buf(pool, buf.into_boxed_slice(), inner)
    }

    /// Creates and returns a new `BufWriter` with `buf` as the internal buffer.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufWriter;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(vec![]);
    /// let pool = CpuPool::new(1);
    /// let buf = vec![0; 4096].into_boxed_slice();
    /// let writer = BufWriter::with_pool_and_buf(pool, buf, f);
    /// # }
    /// ```
    pub fn with_pool_and_buf(pool: CpuPool, buf: Box<[u8]>, inner: W) -> BufWriter<W> {
        BufWriter {
            writer: BufWriterInner {
                inner: inner,
                buf: buf,
                pos: 0,
                w_start: 0,
            },
            pool: pool,
        }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// It is likely invalid to read directly from the underlying writer and then use the
    /// `BufWriter` again.
    pub fn get_ref(&self) -> &W {
        &self.writer.inner
    }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// It is likely invalid to read directly from the underlying writer and then use the
    /// `BufWriter` again.
    pub fn get_mut(&mut self) -> &W {
        &mut self.writer.inner
    }

    /// Returns the internal components of a `BufWriter`, allowing reuse. This
    /// is unsafe because it does not zero the memory of the buffer, meaning
    /// the buffer could countain uninitialized memory.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufWriter;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(b"foo text".to_vec());
    /// let pool = CpuPool::new(1);
    /// let writer = BufWriter::with_pool_and_capacity(pool, 4<<10, f);
    ///
    /// let (f, buf, pool) = unsafe { writer.components() };
    /// assert_eq!(f.get_ref(), b"foo text");
    /// assert_eq!(buf.len(), 4<<10);
    /// # }
    /// ```
    pub unsafe fn components(self) -> (W, Box<[u8]>, CpuPool) {
        let BufWriter {
            writer: BufWriterInner { inner: w, buf, .. },
            pool,
            ..
        } = self;
        (w, buf, pool)
    }

    /// Sets the `BufWriter`s internal buffer position to `pos`.
    ///
    ///
    /// This is _highly_ unsafe for the following reasons:
    ///
    ///   - the internal buffer may have uninitialized memory, and advancing the write position
    ///     will mean writing uninitialized memory
    ///
    ///   - the pos is not validated, meaning it is possible to move the pos past the end of the
    ///     internal buffer. This will cause a panic on the next use of `write_all` or
    ///     `flush_buf`.
    ///
    ///   - it is possible to move _before_ the internal beginning write position, meaning a write
    ///     or flush may panic due to the beginning of a write being after the end of a write.
    ///
    /// This function should only be used for setting up a new `BufWriter` with a buffer that
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
    /// # use futures_bufio::BufWriter;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(vec![]);
    /// let pool = CpuPool::new(1);
    /// let mut buf = vec![0; 4096].into_boxed_slice();
    ///
    /// // copy some known text to uor buffer - note it must be at the beginning
    /// let p = b"pre-existing text";
    /// &mut buf[0..p.len()].copy_from_slice(p);
    ///
    /// let mut writer = BufWriter::with_pool_and_buf(pool, buf, f);
    ///
    /// // unsafely move the writer's position to the end of our known text, and flush the buffer
    /// unsafe { writer.set_pos(p.len()); }
    /// let writer = writer.flush_buf().wait().unwrap_or_else(|(_, e)| {
    ///     panic!("unable to flush_buf: {}", e);
    /// });
    ///
    /// // the underlying writer should be the contents of p
    /// let (f, _, _) = unsafe { writer.components() };
    /// assert_eq!(f.get_ref().as_slice(), p);
    /// # }
    /// ```
    pub unsafe fn set_pos(&mut self, pos: usize) {
        self.writer.pos = pos;
    }

    /// Writes all of `buf` to the `BufWriter`, returning the buffer for potential reuse and any
    /// error that occurs.
    ///
    /// If used on `io::File`'s, `BufWriter` could be valuable for performing page-aligned writes.
    ///
    /// # Examples
    /// ```
    /// # extern crate futures_cpupool;
    /// # extern crate futures;
    /// # extern crate futures_bufio;
    /// #
    /// # use futures::Future;
    /// # use futures_cpupool::CpuPool;
    /// # use futures_bufio::BufWriter;
    /// # use std::io;
    /// # fn main() {
    /// let f = io::Cursor::new(vec![]);
    /// let pool = CpuPool::new(1);
    /// let writer = BufWriter::with_pool_and_capacity(pool, 4096, f);
    ///
    /// let buf = b"many small writes".to_vec();
    /// let (writer, buf) = writer.write_all(buf).wait().unwrap_or_else(|(_, _, e)| {
    ///     // in real usage, we have the option to deconstruct our BufWriter or reuse buf here
    ///     panic!("unable to read full: {}", e);
    /// });
    /// assert_eq!(&*buf, b"many small writes"); // we can reuse buf
    /// # }
    /// ```
    pub fn write_all<B>(
        mut self,
        buf: B,
    ) -> impl Future<Item = OkWrite<Self, B>, Error = ErrWrite<Self, B>>
    where
        B: Deref<Target = [u8]> + Send + 'static,
    {
        let mut rem = buf.len();
        let mut at = 0;
        let mut write_buf = false;

        if self.writer.pos == 0 {
            if buf.len() < self.writer.buf.len() {
                self.writer.pos = copy(&mut self.writer.buf, &*buf);
                return Either::A(ok::<OkWrite<Self, B>, ErrWrite<Self, B>>((self, buf)));
            }
        } else {
            at = copy(&mut self.writer.buf[self.writer.pos..], &*buf);
            self.writer.pos += at;
            rem -= at;

            if self.writer.pos != self.writer.buf.len() {
                return Either::A(ok::<OkWrite<Self, B>, ErrWrite<Self, B>>((self, buf)));
            }
            write_buf = true;
        }

        let BufWriter { mut writer, pool } = self;

        let fut = pool.spawn_fn(move || {
            if write_buf {
                if let Err(e) = writer.inner.write_all(&writer.buf[writer.w_start..]) {
                    return Err((writer, buf, e));
                }
                writer.w_start = 0;
            }

            if rem >= writer.buf.len() {
                let n_write = rem -
                    if writer.buf.len() != 0 {
                        rem % writer.buf.len()
                    } else {
                        0
                    };
                if let Err(e) = writer.inner.write_all(&buf[at..at + n_write]) {
                    return Err((writer, buf, e));
                }
                at += n_write;
                rem -= n_write;
            }

            writer.pos = copy(&mut writer.buf, &buf[at..]);
            Ok((writer, buf))
        });

        Either::B(fut.then(|res| match res {
            Ok((writer, buf)) => Ok((BufWriter { writer, pool }, buf)),
            Err((writer, buf, e)) => Err((BufWriter { writer, pool }, buf, e)),
        }))
    }

    /// Flushes currently buffered data to the inner writer.
    ///
    /// Calling `flush_buf` does not empty the current buffer; instead, a future flush triggered
    /// from `write_all` will be shorter. This is done to keep the full-buffered writes aligned.
    ///
    /// # Examples
    /// ```ignore
    /// let future = writer.flush_buf();
    /// ```
    pub fn flush_buf(self) -> impl Future<Item = Self, Error = (Self, io::Error)> {
        if self.writer.w_start == self.writer.pos {
            return Either::A(ok::<Self, (Self, io::Error)>(self));
        }

        let BufWriter { mut writer, pool } = self;

        let fut = pool.spawn_fn(move || {
            if let Err(e) = writer.inner.write_all(
                &writer.buf[writer.w_start..writer.pos],
            )
            {
                return Err((writer, e));
            }
            writer.w_start = writer.pos;
            Ok(writer)
        });

        Either::B(fut.then(|res| match res {
            Ok(writer) => Ok(BufWriter { writer, pool }),
            Err((writer, e)) => Err((BufWriter { writer, pool }, e)),
        }))
    }

    /// Calls [`flush`] on the inner writer.
    ///
    /// [`flush`]: https://doc.rust-lang.org/std/io/trait.Write.html#tymethod.flush
    ///
    /// # Examples
    /// ```ignore
    /// let future = writer.flush_inner();
    /// ```
    pub fn flush_inner(self) -> impl Future<Item = Self, Error = (Self, io::Error)> {
        let BufWriter { mut writer, pool } = self;
        let fut = pool.spawn_fn(move || {
            if let Err(e) = writer.inner.flush() {
                return Err((writer, e));
            }
            Ok(writer)
        });

        fut.then(|res| match res {
            Ok(writer) => Ok(BufWriter { writer, pool }),
            Err((writer, e)) => Err((BufWriter { writer, pool }, e)),
        })
    }
}

#[test]
fn test_write() {
    use std::fs;
    use std::io::Read;

    fn assert_foo(exp: &'static str) {
        let mut foo = fs::File::open("foo.txt").expect("re-open");
        let mut contents = String::new();
        foo.read_to_string(&mut contents).expect(
            "unable to read file",
        );
        assert_eq!(contents, exp);
    }

    let f = BufWriter::with_pool_and_capacity(
        CpuPool::new(1),
        10,
        fs::OpenOptions::new()
            .write(true)
            .read(true) // for seek
            .create_new(true)
            .open("foo.txt")
            .expect("foo not exclusively created?"),
    );

    // memory (5)
    let (f, buf) = f.write_all(b"hello".to_vec()).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to write to file: {}", e)
        },
    );
    assert_eq!(f.writer.pos, 5);
    assert_eq!(f.writer.w_start, 0);
    assert_eq!(&*buf, b"hello");
    assert_foo("");

    let f = f.flush_buf().wait().unwrap_or_else(|(_, e)| {
        panic!("unable to flush buf: {}", e)
    });
    assert_eq!(f.writer.pos, 5);
    assert_eq!(f.writer.w_start, 5);
    assert_foo("hello");

    let f = f.flush_inner().wait().unwrap_or_else(|(_, e)| {
        panic!("unable to flush file: {}", e)
    });
    assert_eq!(f.writer.pos, 5);
    assert_eq!(f.writer.w_start, 5);
    assert_foo("hello");

    // memory (2) with w_start at 5
    let (f, buf) = f.write_all(b"tw".to_vec()).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to write to file: {}", e)
        },
    );
    assert_eq!(f.writer.pos, 7);
    assert_eq!(f.writer.w_start, 5);
    assert_eq!(&*buf, b"tw");
    assert_foo("hello");

    let f = f.flush_buf().wait().unwrap_or_else(|(_, e)| {
        panic!("unable to flush buf: {}", e)
    });
    assert_eq!(f.writer.pos, 7);
    assert_eq!(f.writer.w_start, 7);
    assert_foo("hellotw");

    // memory (7)
    let (f, buf) = f.write_all(b"goodbye".to_vec()).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to write to file: {}", e)
        },
    );
    assert_eq!(f.writer.pos, 4);
    assert_eq!(f.writer.w_start, 0);
    assert_eq!(&*buf, b"goodbye");
    assert_foo("hellotwgoo");

    // memory (6) + disk (10)
    let (f, buf) = f.write_all(b"more++andthenten".to_vec())
        .wait()
        .unwrap_or_else(|(_, _, e)| panic!("unable to write to file: {}", e));
    assert_eq!(f.writer.pos, 0);
    assert_eq!(f.writer.w_start, 0);
    assert_eq!(&*buf, b"more++andthenten");
    assert_foo("hellotwgoodbyemore++andthenten");

    // disk (10)
    let (f, buf) = f.write_all(b"andtenmore".to_vec()).wait().unwrap_or_else(
        |(_, _, e)| {
            panic!("unable to write to file: {}", e)
        },
    );
    assert_eq!(f.writer.pos, 0);
    assert_eq!(f.writer.w_start, 0);
    assert_eq!(&*buf, b"andtenmore");
    assert_foo("hellotwgoodbyemore++andthentenandtenmore");

    // disk (10) + mem (5)
    let (f, buf) = f.write_all(b"this is rly old".to_vec())
        .wait()
        .unwrap_or_else(|(_, _, e)| panic!("unable to write to file: {}", e));
    assert_eq!(f.writer.pos, 5);
    assert_eq!(f.writer.w_start, 0);
    assert_eq!(&*buf, b"this is rly old");
    assert_foo("hellotwgoodbyemore++andthentenandtenmorethis is rl");

    let f = f.flush_buf().wait().unwrap_or_else(|(_, e)| {
        panic!("unable to flush buf: {}", e)
    });
    assert_eq!(f.writer.pos, 5);
    assert_eq!(f.writer.w_start, 5);
    assert_foo("hellotwgoodbyemore++andthentenandtenmorethis is rly old");

    fs::remove_file("foo.txt").expect("expected file to be removed");
}

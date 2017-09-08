/// A common error string that we use whenever expecting a `read()` call on a `&[u8]`.
pub(crate) const U8READ: &str = "&[u8] reads never error";

/// Copies as many `T` as possible from `src` into `dst`, returning the number of `T` copied. This
/// function is short form for `dst.copy_from_slice(src)`, but accounts for if their lengths are
/// unequal to avoid panics.
///
/// # Examples
///
/// ```ignore
/// let mut l = b"hello".to_vec();
/// let r = b"goodbye".to_vec();
///
/// assert_eq!(copy(&mut l, &r), 5);
/// assert_eq!(l, b"goodb");
/// ```
#[inline]
pub(crate) fn copy(dst: &mut [u8], src: &[u8]) -> usize {
    // Note that Read is implemented for &[u8] and special cases length-1
    // slices to be faster than copy_from_slice.
    use std::io::Read;
    let mut src = src;
    let src_len = src.len();
    let dst_len = dst.len();
    if dst_len >= src_len {
        src.read(&mut dst[..src_len]).expect(U8READ)
    } else {
        (&src[..dst_len]).read(dst).expect(U8READ)
    }
}

#[test]
fn test_copy() {
    fn lr() -> (Vec<u8>, Vec<u8>) {
        (b"hello".to_vec(), b"goodbye".to_vec())
    }

    // longer to shorter
    let (mut l, r) = lr();
    assert_eq!(copy(&mut l, &r), 5);
    assert_eq!(l, b"goodb");
    assert_eq!(r, b"goodbye");

    // shorter to longer
    let (l, mut r) = lr();
    assert_eq!(copy(&mut r, &l[..4]), 4);
    assert_eq!(l, b"hello");
    assert_eq!(r, b"hellbye");

    // dst length 0
    let (mut l, r) = lr();
    assert_eq!(copy(&mut l[..0], &r), 0);
    assert_eq!(l, b"hello");
    assert_eq!(r, b"goodbye");

    // src length 0
    assert_eq!(copy(&mut l, &r[..0]), 0);
    assert_eq!(l, b"hello");
    assert_eq!(r, b"goodbye");
}

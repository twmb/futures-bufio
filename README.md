futures-bufio
=============

[![Build Status](https://travis-ci.org/twmb/futures-bufio.svg?branch=master)](https://travis-ci.org/twmb/futures-bufio)  [![Crates.io](https://img.shields.io/crates/v/futures-bufio.svg)](https://crates.io/crates/futures-bufio) [![Documentation](https://docs.rs/futures-bufio/badge.svg)](https://docs.rs/futures-bufio/)

Buffered IO with futures on top of a threadpool for blocking IO. This crate is
primarily useful for readers or writers that cannot return EWOULDBLOCK, but may
block or sleep (i.e., file IO).

This uses the nightly only feature [`conservative_impl_trait`](https://doc.rust-lang.org/nightly/unstable-book/language-features/conservative-impl-trait.html)
to make the types on non-allocating futures understable.

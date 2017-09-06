//! This library provides buffered IO with futures on top of a threadpool for blocking IO.
//!
//! This crate is most useful when readers or writers do not or cannot block, but do put threads to
//! sleep. For example, files can always read or write, but their reads or writes are slow.
//!
//! This crate uses the nightly-only feature [`conservative_impl_trait`] to eliminate box
//! allocations around futures while still making the return types semi-readable.
//!
//! [`conservative_impl_trait`]: https://doc.rust-lang.org/nightly/unstable-book/language-features/conservative-impl-trait.html
#![feature(conservative_impl_trait)]

extern crate futures;
extern crate futures_cpupool;

mod read;
//mod write;
//mod common;

pub use read::*;
//pub use write::*;

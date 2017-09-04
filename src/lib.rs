//! This library provides buffered IO with futures on top of a threadpool for blocking IO.
//!
//! This crate is most useful when readers or writers do not or cannot block, but do put threads to
//! sleep. For example, files can always read or write, but their reads or writes are slow.
//!
//! This crate uses three nightly-only features: [`proc_macro`], [`conservative_impl_trait`], and
//! [`generators`]. These requirements come from using [`futures-await`]. It is possible to write
//! all code in this crate with explicit (non impl) futures as return types and to not use
//! async/await, but the code would be much, much uglier.
//!
//! [`futures-await`]: https://github.com/alexcrichton/futures-await
//! [`proc_macro`]: https://doc.rust-lang.org/nightly/unstable-book/language-features/proc-macro.html
//! [`conservative_impl_trait`]: https://doc.rust-lang.org/nightly/unstable-book/language-features/conservative-impl-trait.html
//! [`generators`]: https://doc.rust-lang.org/nightly/unstable-book/language-features/generators.html
#![feature(proc_macro, conservative_impl_trait, generators)]

extern crate futures_await as futures;
extern crate futures_cpupool;

mod read;
mod write;
mod common;

pub use read::*;
pub use write::*;

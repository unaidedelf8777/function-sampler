// Lints to detect any obvious fixable perf issues which can be easily fixed.
// because everyones favourite thing is the compiler whining at them.
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::redundant_clone)]
#![warn(suspicious_double_ref_op)]
#![warn(clippy::redundant_clone)]
#![warn(clippy::inefficient_to_string)]
#![warn(clippy::box_collection)]
#![warn(clippy::manual_memcpy)]
#![warn(clippy::large_digit_groups)]
#![warn(clippy::large_enum_variant)]
#![warn(clippy::vec_box)]
#![warn(clippy::unnecessary_cast)]
#![warn(clippy::implicit_clone)]


#[macro_use]
extern crate lazy_static;

// did I mention I like performance?
extern crate mimalloc;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod lazy_index;
mod tokenizer_index;
mod types;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use tokenizer_index::create_fsm_index_end_to_end_py;

use crate::lazy_index::LazyFSMIndex;

#[cfg(feature = "e2e_experimental")]
use crate::tokenizer_index::pattern_to_token_subsets_py;

#[pymodule]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;

    #[cfg(feature = "e2e_experimental")]
    m.add_function(wrap_pyfunction!(pattern_to_token_subsets_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;

    Ok(())
}

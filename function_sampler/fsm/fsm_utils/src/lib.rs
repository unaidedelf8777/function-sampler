#[macro_use]
extern crate lazy_static;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use tokenizer_index::create_fsm_index_end_to_end_py;

mod lazy_index;
mod tokenizer_index;
mod types;

use lazy_index::LazyFSMIndex;

#[pymodule]
#[pyo3(name = "fsm_utils")]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;
    Ok(())
}

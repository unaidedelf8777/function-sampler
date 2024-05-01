#[macro_use]
extern crate lazy_static;

mod lazy_index;
mod tokenizer_index;
mod types;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use tokenizer_index::create_fsm_index_end_to_end_py;

use crate::{lazy_index::LazyFSMIndex};

#[cfg(feature = "e2e_experimental")]
use crate::tokenizer_index::pattern_to_token_subsets_py;

#[pymodule]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;

    #[cfg( feature = "e2e_experimental" )]
    m.add_function(wrap_pyfunction!(pattern_to_token_subsets_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;

    Ok(())
}

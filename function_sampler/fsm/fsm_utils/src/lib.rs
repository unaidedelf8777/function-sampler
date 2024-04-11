use tokenizer_index::create_fsm_index_end_to_end_py;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod tokenizer_index;

#[pymodule]
#[pyo3(name="fsm_utils")]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;
    Ok(())
}

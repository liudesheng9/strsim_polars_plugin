use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub(super) fn native_damerau_levenshtein(a: &str, b: &str) -> i64 {
    strsim::damerau_levenshtein(a, b) as i64
}

// Workaround for arrow::ffi module resolution issue
mod arrow {
    pub use polars_arrow::ffi;
}

#[polars_expr(output_type=Int64)]
fn damerau_levenshtein(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    let out: Int64Chunked = arity::binary_elementwise_values(a, b, native_damerau_levenshtein);
    Ok(out.into_series())
}

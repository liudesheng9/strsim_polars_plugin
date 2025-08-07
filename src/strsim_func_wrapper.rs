use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::derive::CallerContext;
use pyo3_polars::export::polars_core::POOL;
use rayon::prelude::*;

pub(super) fn native_damerau_levenshtein(a: &str, b: &str) -> i64 {
    strsim::damerau_levenshtein(a, b) as i64
}

pub(super) fn native_normalized_damerau_levenshtein(a: &str, b: &str) -> f64 {
    strsim::normalized_damerau_levenshtein(a, b) as f64
}

fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (offset, len)
            })
            .collect()
    }
}

pub fn parallel_apply_damerau_levenshtein(
    inputs: &[Series],
    context: CallerContext,
) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    if a.len() != b.len() {
        return Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length, or one of them must be a Utf8 literal.".into(),
        ));
    }
    if context.parallel() {
        let out: Int64Chunked = arity::binary_elementwise_values(a, b, native_damerau_levenshtein);
        Ok(out.into_series())
    } else {
        POOL.install(|| {
            let splits = split_offsets(a.len(), POOL.current_num_threads());

            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let out: Int64Chunked = {
                        let a = a.slice(offset as i64, len);
                        let b = b.slice(offset as i64, len);
                        arity::binary_elementwise_values(&a, &b, |a, b| {
                            native_damerau_levenshtein(a, b)
                        })
                    };
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            Ok(
                Int64Chunked::from_chunk_iter("".into(), chunks.into_iter().flatten())
                    .into_series(),
            )
        })
    }
}

pub fn parallel_apply_normalized_damerau_levenshtein(
    inputs: &[Series],
    context: CallerContext,
) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    if a.len() != b.len() {
        return Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length, or one of them must be a Utf8 literal.".into(),
        ));
    }
    if context.parallel() {
        let out: Float64Chunked =
            arity::binary_elementwise_values(a, b, native_normalized_damerau_levenshtein);
        Ok(out.into_series())
    } else {
        POOL.install(|| {
            let splits = split_offsets(a.len(), POOL.current_num_threads());

            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let out: Float64Chunked = {
                        let a = a.slice(offset as i64, len);
                        let b = b.slice(offset as i64, len);
                        arity::binary_elementwise_values(&a, &b, |a, b| {
                            native_normalized_damerau_levenshtein(a, b)
                        })
                    };
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            Ok(
                Float64Chunked::from_chunk_iter("".into(), chunks.into_iter().flatten())
                    .into_series(),
            )
        })
    }
}

// Workaround for arrow::ffi module resolution issue
mod arrow {
    pub use polars_arrow::ffi;
}

#[polars_expr(output_type=Int64)]
fn damerau_levenshtein(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    parallel_apply_damerau_levenshtein(inputs, context)
}

#[polars_expr(output_type=Float64)]
fn normalized_damerau_levenshtein(
    inputs: &[Series],
    context: CallerContext,
) -> PolarsResult<Series> {
    parallel_apply_normalized_damerau_levenshtein(inputs, context)
}

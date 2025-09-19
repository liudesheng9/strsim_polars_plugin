use crate::apply_utils::parallel_apply;
use crate::weighted_DL;
use polars::prelude::*;
use polars_core::datatypes::{Float64Type, Int64Type};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::derive::CallerContext;
use pyo3_polars::export::polars_core::POOL;
use rayon::prelude::*;
use serde::Deserialize;

pub(super) fn native_damerau_levenshtein(a: &str, b: &str) -> i64 {
    strsim::damerau_levenshtein(a, b) as i64
}

pub(super) fn native_normalized_damerau_levenshtein(a: &str, b: &str) -> f64 {
    let count_a = a.chars().count();
    let count_b = b.chars().count();

    if count_a == 0 || count_b == 0 {
        return 0.0;
    }

    strsim::normalized_damerau_levenshtein(a, b) as f64
}

fn default_wgr() -> f64 {
    1.0
}
#[derive(Deserialize)]
pub struct WeightedDLKwargs {
    #[serde(default = "default_wgr")]
    weighted_geometric_ratio: f64,
}

pub(super) fn native_geometric_weighted_damerau_levenshtein(
    a: &str,
    b: &str,
    weighted_geometric_ratio: f64,
) -> f64 {
    weighted_DL::normalized_descending_weighted_damerau_levenshtein(a, b, weighted_geometric_ratio)
        as f64
}

fn get_all_substrings<'a>(s: &'a str, k: usize) -> Result<Vec<&'a str>, String> {
    if k == 0 {
        return Err("k must be greater than 0".to_string());
    }

    let char_count = s.chars().count();
    if char_count < k {
        return Err("longer string must be longer than k".to_string());
    }

    let mut indices: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    indices.push(s.len());

    let mut result: Vec<&'a str> = Vec::with_capacity(char_count - k + 1);
    for i in 0..(char_count - k + 1) {
        let start = indices[i];
        let end = indices[i + k];
        result.push(&s[start..end]);
    }

    if result.is_empty() {
        return Err("no substrings found".to_string());
    }

    Ok(result)
}

pub(super) fn native_partial_damerau_levenshtein(a: &str, b: &str) -> i64 {
    let count_a = a.chars().count();
    let count_b = b.chars().count();

    if count_a == 0 || count_b == 0 {
        return 0;
    }

    let (shorter, longer, k) = if count_a < count_b {
        (a, b, count_a)
    } else {
        (b, a, count_b)
    };

    let substrings = get_all_substrings(longer, k).unwrap();

    let distances = substrings
        .iter()
        .map(|substring| strsim::damerau_levenshtein(substring, shorter) as i64)
        .collect::<Vec<_>>();

    *distances.iter().min().unwrap()
}

pub(super) fn native_partial_normalized_damerau_levenshtein(a: &str, b: &str) -> f64 {
    let count_a = a.chars().count();
    let count_b = b.chars().count();

    if count_a == 0 || count_b == 0 {
        return 0.0;
    }

    let (shorter, longer, k) = if count_a < count_b {
        (a, b, count_a)
    } else {
        (b, a, count_b)
    };

    let substrings = get_all_substrings(longer, k).unwrap();

    let similarities = substrings
        .iter()
        .map(|substring| strsim::normalized_damerau_levenshtein(substring, shorter))
        .collect::<Vec<_>>();

    *similarities
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
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

pub(super) fn parallel_apply_gwdl(
    inputs: &[Series],
    context: CallerContext,
    kwargs: WeightedDLKwargs,
) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    if a.len() != b.len() {
        return Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length, or one of them must be a Utf8 literal.".into(),
        ));
    }
    if context.parallel() {
        let out: ChunkedArray<Float64Type> = arity::binary_elementwise_values(a, b, |s1, s2| {
            native_geometric_weighted_damerau_levenshtein(s1, s2, kwargs.weighted_geometric_ratio)
        });
        Ok(out.into_series())
    } else {
        POOL.install(|| {
            let splits = split_offsets(a.len(), POOL.current_num_threads());

            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let out: ChunkedArray<Float64Type> = {
                        let a = a.slice(offset as i64, len);
                        let b = b.slice(offset as i64, len);
                        arity::binary_elementwise_values(&a, &b, |a, b| {
                            native_geometric_weighted_damerau_levenshtein(
                                a,
                                b,
                                kwargs.weighted_geometric_ratio,
                            )
                        })
                    };
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            Ok(ChunkedArray::<Float64Type>::from_chunk_iter(
                "".into(),
                chunks.into_iter().flatten(),
            )
            .into_series())
        })
    }
}

// Workaround for arrow::ffi module resolution issue
mod arrow {
    pub use polars_arrow::ffi;
}

#[polars_expr(output_type=Int64)]
fn damerau_levenshtein(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    parallel_apply::<_, Int64Type>(inputs, context, native_damerau_levenshtein)
}

#[polars_expr(output_type=Float64)]
fn normalized_damerau_levenshtein(
    inputs: &[Series],
    context: CallerContext,
) -> PolarsResult<Series> {
    parallel_apply::<_, Float64Type>(inputs, context, native_normalized_damerau_levenshtein)
}

#[polars_expr(output_type=Int64)]
fn partial_damerau_levenshtein(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    parallel_apply::<_, Int64Type>(inputs, context, native_partial_damerau_levenshtein)
}

#[polars_expr(output_type=Float64)]
fn partial_normalized_damerau_levenshtein(
    inputs: &[Series],
    context: CallerContext,
) -> PolarsResult<Series> {
    parallel_apply::<_, Float64Type>(
        inputs,
        context,
        native_partial_normalized_damerau_levenshtein,
    )
}

#[polars_expr(output_type=Float64)]
fn geometric_weighted_damerau_levenshtein(
    inputs: &[Series],
    context: CallerContext,
    kwargs: WeightedDLKwargs,
) -> PolarsResult<Series> {
    parallel_apply_gwdl(inputs, context, kwargs)
}

use crate::apply_utils::parallel_apply;
use crate::weighted_DL;
use polars::prelude::*;
use polars_core::datatypes::{Float64Type, Int64Type};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::derive::CallerContext;

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

#[derive(Deserialize)]
pub struct WeightedDLKwargs {
    #[serde(default = "default_weighted_geometric_ratio")]
    weighted_geometric_ratio: f64,
    #[serde(default = "default_normalized")]
    normalized: bool,
}

#[derive(Deserialize)]
pub struct WeightedDLByWordsKwargs {
    #[serde(default = "default_weighted_geometric_ratio")]
    weighted_geometric_ratio: f64,
    #[serde(default = "default_normalized")]
    normalized: bool,
    #[serde(default = "default_agg")]
    agg: String,
}

fn default_weighted_geometric_ratio() -> f64 {
    1.0
}

fn default_normalized() -> bool {
    false
}

fn default_agg() -> String {
    "mean".to_string()
}

pub(super) fn native_geometric_weighted_damerau_levenshtein(
    a: &str,
    b: &str,
    weighted_geometric_ratio: f64,
    normalized: bool,
) -> f64 {
    weighted_DL::normalized_descending_weighted_damerau_levenshtein(
        a,
        b,
        weighted_geometric_ratio,
        normalized,
    ) as f64
}

pub(super) fn native_geometric_weighted_damerau_levenshtein_bywords(
    a: &str,
    b: &str,
    weighted_geometric_ratio: f64,
    normalized: bool,
    agg: &str,
) -> f64 {
    let agg_method = match agg {
        "max" => weighted_DL::ByWordsAggregation::Max,
        "min" => weighted_DL::ByWordsAggregation::Min,
        _ => weighted_DL::ByWordsAggregation::Mean,
    };
    weighted_DL::normalized_descending_weighted_damerau_levenshtein_bywords(
        a,
        b,
        weighted_geometric_ratio,
        normalized,
        agg_method,
    )
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

pub(super) fn parallel_apply_gwdl(
    inputs: &[Series],
    context: CallerContext,
    kwargs: WeightedDLKwargs,
) -> PolarsResult<Series> {
    let weighted_geometric_ratio = kwargs.weighted_geometric_ratio;
    let normalized = kwargs.normalized;
    parallel_apply::<_, Float64Type>(inputs, context, move |s1, s2| {
        native_geometric_weighted_damerau_levenshtein(s1, s2, weighted_geometric_ratio, normalized)
    })
}

pub(super) fn parallel_apply_gwdl_bywords(
    inputs: &[Series],
    context: CallerContext,
    kwargs: WeightedDLByWordsKwargs,
) -> PolarsResult<Series> {
    let weighted_geometric_ratio = kwargs.weighted_geometric_ratio;
    let normalized = kwargs.normalized;
    let agg = kwargs.agg;
    parallel_apply::<_, Float64Type>(inputs, context, move |s1, s2| {
        native_geometric_weighted_damerau_levenshtein_bywords(
            s1,
            s2,
            weighted_geometric_ratio,
            normalized,
            &agg,
        )
    })
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

#[polars_expr(output_type=Float64)]
fn geometric_weighted_damerau_levenshtein_bywords(
    inputs: &[Series],
    context: CallerContext,
    kwargs: WeightedDLByWordsKwargs,
) -> PolarsResult<Series> {
    parallel_apply_gwdl_bywords(inputs, context, kwargs)
}

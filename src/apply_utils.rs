use polars::prelude::*;
use pyo3_polars::derive::CallerContext;
use pyo3_polars::export::polars_core::POOL;
use rayon::prelude::*;

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

pub fn parallel_apply<F, Out>(
    inputs: &[Series],
    context: CallerContext,
    native_fn: F,
) -> PolarsResult<Series>
where
    F: Fn(&str, &str) -> Out::Native + Sync + Send,
    Out: PolarsNumericType,
{
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    if a.len() != b.len() {
        return Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length, or one of them must be a Utf8 literal.".into(),
        ));
    }
    if context.parallel() {
        let out: ChunkedArray<Out> =
            arity::binary_elementwise_values(a, b, |s1, s2| native_fn(s1, s2));
        Ok(out.into_series())
    } else {
        POOL.install(|| {
            let splits = split_offsets(a.len(), POOL.current_num_threads());

            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let out: ChunkedArray<Out> = {
                        let a = a.slice(offset as i64, len);
                        let b = b.slice(offset as i64, len);
                        arity::binary_elementwise_values(&a, &b, |a, b| native_fn(a, b))
                    };
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            Ok(
                ChunkedArray::<Out>::from_chunk_iter("".into(), chunks.into_iter().flatten())
                    .into_series(),
            )
        })
    }
}

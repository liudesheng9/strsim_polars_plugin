use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone, Copy)]
pub enum ByWordsAggregation {
    Max,
    Mean,
    Min,
}

/* Returns the final index for a value in a single vector that represents a fixed
grid */
fn flat_index(i: usize, j: usize, width: usize) -> usize {
    j * width + i
}

/// Like optimal string alignment, but substrings can be edited an unlimited
/// number of times, and the triangle inequality holds. Weighted version where
/// deletion costs from `a_elems` are multiplied by `weight_a` (position-dependent)
/// and insertion costs to `b_elems` are multiplied by `weight_b` (position-dependent).
/// Substitution cost is the sum of corresponding weights if mismatch. Transposition
/// includes weighted costs for intervening deletions/insertions plus the mean of the
/// four involved character weights: `weight_a` for both swapped chars and `weight_b`
/// for both swapped chars. For an adjacent swap at (i,j), this is
/// `(weight_a[i-1] + weight_a[i-2] + weight_b[j-1] + weight_b[j-2]) / 4`.
pub fn generic_weighted_damerau_levenshtein<Elem>(
    a_elems: &[Elem],
    b_elems: &[Elem],
    weight_a: &[f64],
    weight_b: &[f64],
) -> f64
where
    Elem: Eq + Hash + Clone,
{
    let a_len = a_elems.len();
    let b_len = b_elems.len();

    assert_eq!(weight_a.len(), a_len);
    assert_eq!(weight_b.len(), b_len);

    let mut prefix_a: Vec<f64> = vec![0.0];
    for &w in weight_a {
        prefix_a.push(*prefix_a.last().unwrap() + w);
    }
    let mut prefix_b: Vec<f64> = vec![0.0];
    for &w in weight_b {
        prefix_b.push(*prefix_b.last().unwrap() + w);
    }

    if a_len == 0 {
        return prefix_b[b_len];
    }
    if b_len == 0 {
        return prefix_a[a_len];
    }

    let width = a_len + 2;
    let mut distances = vec![0.0_f64; (a_len + 2) * (b_len + 2)];
    let max_distance = prefix_a[a_len] + prefix_b[b_len] + 1.0;

    distances[0] = max_distance;

    for i in 0..=a_len {
        distances[flat_index(i + 1, 0, width)] = max_distance;
        distances[flat_index(i + 1, 1, width)] = prefix_a[i];
    }

    for j in 0..=b_len {
        distances[flat_index(0, j + 1, width)] = max_distance;
        distances[flat_index(1, j + 1, width)] = prefix_b[j];
    }

    let mut elems: HashMap<Elem, usize> = HashMap::with_capacity(64);

    for i in 1..=a_len {
        let mut db = 0;

        for j in 1..=b_len {
            let k = *elems.get(&b_elems[j - 1]).unwrap_or(&0);

            let deletion_cost_code = distances[flat_index(i, j + 1, width)] + weight_a[i - 1];
            let insertion_cost_code = distances[flat_index(i + 1, j, width)] + weight_b[j - 1];

            let is_match = a_elems[i - 1] == b_elems[j - 1];
            // Substitution uses the maximum of the two position-dependent weights
            // so it is comparable to a single deletion or insertion when weights match.
            let substitution_cost = distances[flat_index(i, j, width)]
                + if is_match {
                    0.0
                } else {
                    weight_a[i - 1].max(weight_b[j - 1])
                };

            let del_between = prefix_a[i - 1] - prefix_a[k];
            let ins_between = prefix_b[j - 1] - prefix_b[db];
            // Transposition base uses the average of the two positions' max weights.
            // This keeps a single swap comparable to a single substitution when the
            // same positions are involved.
            let swap_base = if k > 0 && db > 0 {
                let left_max = weight_a[i - 1].max(weight_b[j - 1]);
                let right_max = weight_a[k - 1].max(weight_b[db - 1]);
                (left_max + right_max) / 2.0
            } else {
                weight_a[i - 1].max(weight_b[j - 1])
            };
            let transposition_cost =
                distances[flat_index(k, db, width)] + del_between + ins_between + swap_base;

            let val = substitution_cost
                .min(deletion_cost_code)
                .min(insertion_cost_code)
                .min(transposition_cost);

            distances[flat_index(i + 1, j + 1, width)] = val;

            if is_match {
                db = j;
            }
        }

        elems.insert(a_elems[i - 1].clone(), i);
    }

    distances[flat_index(a_len + 1, b_len + 1, width)]
}

// weighted damerau levenshtein

/// Generate a descending geometric weight sequence of length `n` with ratio `k`,
/// normalized so the weights sum exactly to `n`.
///
/// Notes:
/// - If `k == 1.0`, all weights are `1`.
/// - If `k > 1.0`, we invert it (use `1/k`) to keep the sequence descending.
/// - Panics if `k <= 0.0`.
fn normalized_geometric_descending_weights(n: usize, k: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    assert!(k > 0.0, "Geometric ratio k must be positive");

    if (k - 1.0).abs() < f64::EPSILON {
        // Equal weights that sum to n
        return vec![1.0; n];
    }

    let ratio = if k > 1.0 { 1.0 / k } else { k };

    // Build raw geometric sequence and accumulate sum in one pass
    let mut weights: Vec<f64> = Vec::with_capacity(n);
    let mut current = 1.0_f64;
    let mut sum = 0.0_f64;
    for _ in 0..n {
        weights.push(current);
        sum += current;
        current *= ratio;
    }

    // Scale so the sum equals n (within floating-point precision)
    let scale = (n as f64) / sum;
    for w in &mut weights {
        *w *= scale;
    }
    weights
}

/// Wrapper over generic weighted Damerau-Levenshtein that uses normalized
/// descending geometric weights for both strings, parameterized by `k`.
///
/// To keep relative costs comparable across different string lengths, we
/// normalize using a shared scale based on `max(len(a), len(b))` and then
/// slice the weights for each string. This avoids making early-character
/// weights larger solely because one string is longer.
pub fn normalized_descending_weighted_damerau_levenshtein(
    a: &str,
    b: &str,
    k: f64,
    normalized: bool,
) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let max_len = a_chars.len().max(b_chars.len());
    let shared_weights = normalized_geometric_descending_weights(max_len, k);
    let weight_a = shared_weights[0..a_chars.len()].to_vec();
    let weight_b = shared_weights[0..b_chars.len()].to_vec();
    match normalized {
        true => {
            let result =
                generic_weighted_damerau_levenshtein(&a_chars, &b_chars, &weight_a, &weight_b);
            result / max_len as f64
        }
        false => generic_weighted_damerau_levenshtein(&a_chars, &b_chars, &weight_a, &weight_b),
    }
}

/// Calculates the Damerau-Levenshtein distance between two strings on a word-by-word basis.
///
/// This function splits the input strings `a` and `b` into words, based on whitespace.
/// It then performs a positional comparison, matching the first word of the shorter list
/// with the first word of the longer list, the second with the second, and so on.
///
/// For each pair of words, it computes the `normalized_descending_weighted_damerau_levenshtein`
/// distance. Finally, it aggregates these distances into a single `f64` value using the
/// method specified by the `agg` parameter.
///
/// # Arguments
///
/// * `a` - The first string.
/// * `b` - The second string.
/// * `k` - The geometric ratio for weighted Damerau-Levenshtein.
/// * `normalized` - If true, the distance for each word pair is normalized by word length.
/// * `agg` - The aggregation method (`Max`, `Mean`, or `Min`) to combine word-level distances.
///
/// # Returns
///
/// A single `f64` representing the aggregated distance. Returns `0.0` if either input
/// string is empty or contains no words.
pub fn normalized_descending_weighted_damerau_levenshtein_bywords(
    a: &str,
    b: &str,
    k: f64,
    normalized: bool,
    agg: ByWordsAggregation,
) -> f64 {
    let a_words: Vec<&str> = a.split_whitespace().collect();
    let b_words: Vec<&str> = b.split_whitespace().collect();

    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    let (shorter, longer) = if a_words.len() < b_words.len() {
        (a_words, b_words)
    } else {
        (b_words, a_words)
    };

    let distances = (0..shorter.len()).map(|i| {
        normalized_descending_weighted_damerau_levenshtein(shorter[i], longer[i], k, normalized)
    });

    match agg {
        ByWordsAggregation::Max => distances.fold(f64::NEG_INFINITY, f64::max),
        ByWordsAggregation::Mean => distances.sum::<f64>() / shorter.len() as f64,
        ByWordsAggregation::Min => distances.fold(f64::INFINITY, f64::min),
    }
}

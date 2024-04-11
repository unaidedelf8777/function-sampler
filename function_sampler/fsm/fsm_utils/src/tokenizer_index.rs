use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, FromPyObject)]
pub struct FSMInfo {
    #[pyo3(item("initial"))]
    initial: i64,
    #[pyo3(item("finals"))]
    finals: HashSet<i64>,
    #[pyo3(item("transitions"))]
    transitions: HashMap<(i64, i64), i64>,
    #[pyo3(item("trans_key_to_states"))]
    trans_key_to_states: HashMap<i64, Vec<i64>>,
    #[pyo3(item("alphabet_anything_value"))]
    alphabet_anything_value: i64,
    #[pyo3(item("alphabet_symbol_mapping"))]
    alphabet_symbol_mapping: HashMap<String, i64>,
}

pub type TokenVocabulary = HashMap<String, Vec<i64>>;

fn walk_fsm(
    fsm_info: &FSMInfo,
    input_string: &str,
    start_state: i64,
    full_match: bool,
) -> Vec<i64> {
    let mut state = start_state;
    let mut accepted_states = Vec::new();
    let mut last_final_idx: Option<usize> = None;

    let mut current_pos = 0;
    let input_chars: Vec<char> = input_string.chars().collect();

    while current_pos < input_chars.len() {
        let mut found = false;

        // Attempt to match longer substrings first, ensuring multi-character sequences are prioritized
        for len in (1..=input_chars.len() - current_pos).rev() {
            let possible_match: String =
                input_chars[current_pos..current_pos + len].iter().collect();

            if let Some(&trans_key) = fsm_info.alphabet_symbol_mapping.get(&possible_match) {
                if let Some(&new_state) = fsm_info.transitions.get(&(state, trans_key)) {
                    state = new_state;
                    if fsm_info.finals.contains(&state) {
                        last_final_idx = Some(accepted_states.len() + 1);
                    }
                    accepted_states.push(state);
                    current_pos += len; // Move past the matched substring
                    found = true;
                    break;
                }
            }
        }

        if !found {
            if !full_match && last_final_idx.is_some() {
                // Non-full match and we've previously encountered a final state
                return accepted_states
                    .into_iter()
                    .take(last_final_idx.unwrap())
                    .collect();
            } else {
                // No match found, or a full match is required
                return vec![];
            }
        }
    }

    // Full match checks
    if full_match && last_final_idx.map_or(true, |idx| idx != accepted_states.len()) {
        return vec![]; // Full match required but last character didn't result in a final state
    }

    accepted_states
}

fn state_scan_tokens(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    start_state: i64,
) -> HashSet<(i64, i64)> {
    vocabulary
        .par_iter()
        .flat_map(|(token, token_ids)| {
            // For each token, perform the FSM walk in parallel.
            let state_seq = walk_fsm(fsm_info, token, start_state, false);

            if state_seq.len() < token.chars().count() {
                None
            } else {
                Some(
                    token_ids
                        .iter()
                        .map(move |&token_id| (token_id, *state_seq.last().unwrap()))
                        .collect::<Vec<_>>(),
                )
            }
        })
        // Flatten the nested structure into a single collection of pairs.
        .flatten()
        // Collect the results into a HashSet to ensure uniqueness.
        .collect::<HashSet<(i64, i64)>>()
}

fn create_fsm_index_end_to_end(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
) -> HashMap<i64, HashMap<i64, i64>> {
    let mut states_to_token_subsets = HashMap::new();
    let mut seen = HashSet::new();
    let mut next_states = HashSet::new();
    next_states.insert(fsm_info.initial);

    while let Some(start_state) = next_states.iter().next().copied() {
        next_states.remove(&start_state);
        let token_ids_end_states = state_scan_tokens(fsm_info, vocabulary, start_state);

        for &(token_id, end_state) in &token_ids_end_states {
            states_to_token_subsets
                .entry(start_state)
                .or_insert_with(HashMap::new)
                .insert(token_id, end_state);
            if !seen.contains(&end_state) {
                next_states.insert(end_state);
            }
        }

        
        seen.insert(start_state);
    }

    states_to_token_subsets
}

/// Create an FSM state-to-vocabulary map/index through end-to-end token parsing.
///
/// Args:
///     fsm_info (dict): A dictionary containing FSM information.
///     vocabulary (dict): A dictionary representing the vocabulary tokens.
///
/// Returns:
///     dict: A mapping of FSM states to vocabulary token sets.
#[pyfunction(name = "create_fsm_index_end_to_end")]
#[pyo3(text_signature = "(fsm_info, vocabulary, /)")]
pub fn create_fsm_index_end_to_end_py(
    py: Python<'_>,
    fsm_info: FSMInfo,
    vocabulary: TokenVocabulary,
) -> HashMap<i64, HashMap<i64, i64>> {
    let states_to_token_subsets =
        py.allow_threads(move || create_fsm_index_end_to_end(&fsm_info, &vocabulary));

    states_to_token_subsets
}

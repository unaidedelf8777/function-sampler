// third party imports
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};


// local module imports
use crate::types::{build_dfa, FSMInfo};

pub type TokenVocabulary = BTreeMap<String, Vec<u32>>;

fn walk_fsm(
    fsm_info: &FSMInfo,
    input_string: &str,
    start_state: u32,
    full_match: bool,
) -> Vec<u32> {
    
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

#[inline(always)]
fn state_scan_tokens(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    start_state: u32,
) -> BTreeSet<(u32, u32)> {
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
        .collect::<BTreeSet<(u32, u32)>>()
}

fn create_fsm_index_end_to_end(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
) -> BTreeMap<u32, BTreeMap<u32, u32>> {
    let mut states_to_token_subsets = BTreeMap::<u32, BTreeMap<u32, u32>>::new();
    let mut seen = BTreeSet::<u32>::new();
    let mut next_states = BTreeSet::<u32>::new();
    next_states.insert(fsm_info.initial);

    while let Some(start_state) = next_states.iter().next().copied() {
        next_states.remove(&start_state);
        let token_ids_end_states = state_scan_tokens(fsm_info, vocabulary, start_state);

        for &(token_id, end_state) in &token_ids_end_states {
            states_to_token_subsets
                .entry(start_state)
                .or_insert_with(BTreeMap::new)
                .insert(token_id, end_state);
            if seen.insert(end_state) {
                next_states.insert(end_state);
            }
        }

        seen.insert(start_state);
    }

    states_to_token_subsets
}


fn trim_vocabulary(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
) -> TokenVocabulary {
    vocabulary
        .par_iter()
        .filter(|(token, _)| {
            // Check each character or substring in the token
            token.chars().all(|ch| {
                // For simplicity, assuming single character mapping here
                let ch_str = ch.to_string();
                fsm_info.alphabet_symbol_mapping.contains_key(&ch_str)
            })
        })
        .map(|(token, ids)| (token.clone(), ids.clone()))
        .collect()
}

/// Create an FSM state-to-vocabulary map/index through end-to-end token parsing.
///
/// Args:
///     pattern (String): A string pattern to build the DFA.
///     vocabulary (TokenVocabulary): A data structure representing the vocabulary tokens.
///
/// Returns:
///     (BTreeMap<u32, BTreeMap<u32, u32>>, u32, Vec<u32>): A mapping of FSM states to vocabulary token sets,
///     the initial state, and a vector of final states.
#[pyfunction(name = "create_fsm_index_end_to_end")]
#[pyo3(text_signature = "(fsm_info, vocabulary, /)")]
pub fn create_fsm_index_end_to_end_py(
    py: Python<'_>,
    pattern: String,
    vocabulary: TokenVocabulary,
) -> (BTreeMap<u32, BTreeMap<u32, u32>>, u32, Vec<u32>) {
    let start_time = std::time::Instant::now();
    let results = py.allow_threads(move || {
        let dfa = build_dfa(&pattern, false);
        let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());
        println!("initial vocab len: {:?}", vocabulary.len());
        let vocabulary = trim_vocabulary(&fsm_info, &vocabulary);
        println!("final vocab len: {:?}", vocabulary.len());
        let states_token_subsets = create_fsm_index_end_to_end(&fsm_info, &vocabulary);
        println!("Time taken to build FSM index: {:?}", start_time.elapsed());
        println!(
            "length states to token subsets: {:?}",
            states_token_subsets.len()
        );
        (fsm_info, states_token_subsets)
    });

    (
        results.1,
        results.0.initial,
        results.0.finals.iter().copied().collect(),
    )
}

/// Create an FSM state-to-vocabulary map/index through end-to-end token parsing.
/// ( Rust version for tests and other stuff )
/// 
/// Args:
///     pattern (String): A string pattern to build the DFA.
///     vocabulary (TokenVocabulary): A data structure representing the vocabulary tokens.
///
/// Returns:
///     (BTreeMap<u32, BTreeMap<u32, u32>>, u32, Vec<u32>): A mapping of FSM states to vocabulary token sets,
///     the initial state, and a vector of final states.
pub fn create_fsm_index_tokenizer(
    pattern: String,
    vocabulary: TokenVocabulary,
) -> (BTreeMap<u32, BTreeMap<u32, u32>>, u32, Vec<u32>) {
    let start_time = std::time::Instant::now();

    // Building the DFA from the pattern without Python interference
    let dfa = build_dfa(&pattern, false);
    let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());

    // Trimming the vocabulary based on the FSM info
    let trimmed_vocabulary = trim_vocabulary(&fsm_info, &vocabulary);

    // Creating the FSM index end-to-end
    let states_token_subsets = create_fsm_index_end_to_end(&fsm_info, &trimmed_vocabulary);

    (
        states_token_subsets,
        fsm_info.initial,
        fsm_info.finals.iter().copied().collect(),
    )
}

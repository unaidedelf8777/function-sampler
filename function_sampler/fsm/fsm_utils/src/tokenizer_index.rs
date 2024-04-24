/// This module provides utility functions for tokenizing input strings using a finite state machine (FSM).
///
/// The `walk_fsm` function performs a FSM walk on an input string, starting from a given state, and returns a vector of accepted states.
///
/// The `state_scan_tokens` function scans a token vocabulary using the FSM and returns a set of token IDs and their corresponding end states.
///
/// The `create_fsm_index_end_to_end_parallel` function creates an FSM state-to-vocabulary map/index through end-to-end token parsing in parallel.
///
/// The `trim_vocabulary` function trims the token vocabulary by filtering out tokens that contain characters not present in the FSM's alphabet.
///
/// The `create_fsm_index_end_to_end_py` function is a Python interface for creating an FSM state-to-vocabulary map/index through end-to-end token parsing.
/// It takes a pattern string, a token vocabulary, and an end-of-sequence token ID as input, and returns a LazyFSMIndex object.

/// * Imports * ///
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Condvar, Mutex};

use crate::{
    lazy_index::LazyFSMIndex,
    types::{build_dfa, FSMInfo, TokenVocabulary},
};

/// ### `walk_fsm`
///
/// This function performs a walk through a finite state machine (FSM) based on the provided input string starting from a specified state.
///
/// **Arguments:**
/// - `fsm_info`: Reference to `FSMInfo` which holds the details about the FSM, including transitions and final states.
/// - `input_string`: The string input to be tokenized or processed through the FSM.
/// - `start_state`: The initial state from which the FSM walk begins.
/// - `full_match`: A boolean that determines if the function should return results only when the entire input string is matched to a final state.

/// **Returns:**
/// - `Vec<u32>`: A vector of accepted states after processing the input string through the FSM.
/// This vector includes states that are reached which form a part of the final states of the FSM, depending on the `full_match` requirement.
///
/// **Description:**
/// The function iterates over the input string, trying to match larger substrings first to accommodate multi-character transitions in the FSM.
/// If a substring matches and leads to a state that is a final state, it records this position. Depending on the `full_match` flag,
/// it may return early or continue to process until all substrings are attempted. The function is sensitive to the ordering of characters
/// and transitions, ensuring that the longest possible matches are considered first.
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

/// ### `state_scan_tokens` Function
///
/// This function scans a set of tokens against an FSM to determine the resulting states from a given start state.
///
/// **Arguments:**
/// - `fsm_info`: Reference to `FSMInfo` containing FSM transition rules and other metadata.
/// - `vocabulary`: Reference to `TokenVocabulary`, a collection of tokens that are to be tested against the FSM.
/// - `start_state`: The initial state from which token scanning begins in the FSM.
///
/// **Returns:**
/// - `BTreeSet<(u32, u32)>`: A set of tuples where each tuple consists of a token ID and the corresponding end state in the FSM after processing the token.
///
/// **Description:**
/// The function iterates over each token in the vocabulary and applies `walk_fsm` to determine how far the token can be processed within the FSM starting from the `start_state`.
/// If a token can be fully processed (i.e., the length of the state sequence returned by `walk_fsm` matches the token length), the end state and token ID are recorded.
/// The results are unique due to the nature of `BTreeSet`, ensuring no duplicate entries for tokens leading to the same end state.
#[inline(always)]
fn state_scan_tokens(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    start_state: u32,
) -> BTreeSet<(u32, u32)> {
    vocabulary
        .iter()
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

/// ### `create_fsm_index_end_to_end_parallel` Function
///
/// Creates a mapping of FSM states to vocabulary tokens in parallel, facilitating quick construction of state-to-token maps for large vocabularies.
///
/// **Arguments:**
/// - `fsm_info`: An `Arc<FSMInfo>` encapsulating the FSM's transition rules and final states.
/// - `vocabulary`: An `Arc<TokenVocabulary>` representing the set of tokens to be used.
/// - `return_to`: An `Arc<Mutex<BTreeMap<u32, BTreeMap<u32, u32>>>>` where the resulting state-to-token map is stored.
/// - `state_notifiers`: An `Arc<Mutex<BTreeMap<u32, Arc<(Mutex<bool>, Condvar)>>>>` used to notify other threads about the completion of state computations.
///
/// **Returns:**
/// - `None`: The function returns nothing, but it populates `return_to` with the computed state-to-token maps and notifies other processes of completion through `state_notifiers`.
///
/// **Description:**
/// The function processes each state in the FSM in parallel, applying `state_scan_tokens` to build a map from each state to possible tokens and their resultant states.
/// It fills `return_to` with these mappings and uses `state_notifiers` to signal the completion of the computation for each state,
///  enabling efficient multi-threaded computation and synchronization.
pub fn create_fsm_index_end_to_end_parallel(
    fsm_info: Arc<FSMInfo>,
    vocabulary: Arc<TokenVocabulary>,
    return_to: Arc<Mutex<BTreeMap<u32, BTreeMap<u32, u32>>>>,
    state_notifiers: Arc<Mutex<BTreeMap<u32, Arc<(Mutex<bool>, Condvar)>>>>,
) {
    fsm_info.states.par_iter().for_each(|&start_state| {
        let token_ids_end_states = state_scan_tokens(&fsm_info, &vocabulary, start_state);

        let mut map = BTreeMap::new();
        for &(token_id, end_state) in &token_ids_end_states {
            map.insert(token_id, end_state);
        }

        // Lock the mutex to access the map and insert the new state map
        {
            let mut return_to_locked = return_to.lock().unwrap();
            return_to_locked.insert(start_state, map);
        }

        // Retrieve the notifier for the current state and notify all waiting threads
        let notifier = {
            let mut notifiers = state_notifiers.lock().unwrap();
            notifiers
                .entry(start_state)
                .or_insert_with(|| Arc::new((Mutex::new(false), Condvar::new())))
                .clone()
        };

        // Set the state to done and notify all waiters
        let (done_lock, condvar) = &*notifier;
        let mut done = done_lock.lock().unwrap();
        *done = true;
        condvar.notify_all();
    });
}

/// ### `trim_vocabulary` Function
///
/// Filters out tokens from the given vocabulary that contain characters not present in the FSM's alphabet.
///
/// **Arguments:**
/// - `fsm_info`: Reference to `FSMInfo` which includes the allowed alphabet for the FSM.
/// - `vocabulary`: Reference to `TokenVocabulary` which includes the full set of tokens to be filtered.
///
/// **Returns:**
/// - `TokenVocabulary`: A new vocabulary containing only those tokens where every character (or substring) is present in the FSM's alphabet.
///
/// **Description:**
/// This function uses parallel processing to quickly filter out tokens that include any character or substring not found in the FSM's alphabet, 
/// as defined in `fsm_info`. It checks each token and retains those where all characters are valid per the FSM's alphabet mappings,
/// thus ensuring the trimmed vocabulary only contains tokens that can be processed by the FSM.
fn trim_vocabulary(fsm_info: &FSMInfo, vocabulary: &TokenVocabulary) -> TokenVocabulary {
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
    eos_token_id: u32,
) -> LazyFSMIndex {
    let results = py.allow_threads(move || {
        let dfa = build_dfa(&pattern, false);
        let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());
        let trimmed_vocabulary = trim_vocabulary(&fsm_info, &vocabulary);
        return LazyFSMIndex::new(fsm_info, trimmed_vocabulary, eos_token_id);
    });
    return results;
}

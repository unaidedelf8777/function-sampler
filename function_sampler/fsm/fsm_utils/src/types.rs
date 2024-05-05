#[cfg(feature = "e2e_experimental")]
use regex_automata::{
    dfa::{dense::DFA, Automaton, StartKind},
    util::{primitives::StateID, syntax::Config as SyntaxConfig},
    Anchored,
};

use std::sync::Arc;

use std::collections::{BTreeMap, BTreeSet};
/// since pyo3 cant convert to BTreeMap and BTreeSet
use std::collections::{HashMap, HashSet};

use std::convert::TryFrom;

use pyo3::prelude::*;

lazy_static! {
    // very tiny optim so we dont need to iterate over unnesescarry bytes we will never use.
    static ref RANGES: [(u32, u32); 8] = [
        (97, 122),   // Lowercase letters [a-z]
        (65, 90),    // Uppercase letters [A-Z]
        (48, 57),    // Digits [0-9]
        (32, 47),    // Punctuation and symbols (first part)
        (58, 64),    // Punctuation and symbols (second part)
        (91, 96),    // Punctuation and symbols (third part)
        (123, 126),  // Punctuation and symbols (fourth part)
        (9, 10),     // tab, newline ( \t, \n )
    ];
}

/// `TokenVocabulary` is a type alias for a `BTreeMap` where the key is a `char` representing a token, and the value is a `Vec<u32>` containing token IDs. This structure is used to store and manage tokens for processing with finite state machines (FSMs), ensuring ordered access and efficient retrieval without hashing overhead.
pub type TokenVocabulary = BTreeMap<String, Vec<u32>>;

#[derive(Debug, Clone, FromPyObject)]
pub struct PyFSMInfo {
    #[pyo3(item("initial"))]
    initial: u32,
    #[pyo3(item("finals"))]
    finals: HashSet<u32>,
    #[pyo3(item("transitions"))]
    transitions: HashMap<(u32, u32), u32>,
    //#[pyo3(item("trans_key_to_states"))]
    //trans_key_to_states: HashMap<u32, Vec<u32>>,
    //#[pyo3(item("alphabet_anything_value"))]
    //alphabet_anything_value: u32,
    #[pyo3(item("alphabet_symbol_mapping"))]
    alphabet_symbol_mapping: HashMap<String, u32>,
}

/// For mapping PyFSMInfo to std FSMInfo object.

impl TryFrom<&PyFSMInfo> for FSMInfo {
    type Error = &'static str; // Simplify error handling for this example

    fn try_from(py_info: &PyFSMInfo) -> Result<Self, Self::Error> {
        // Direct assignment as the type is assumed to be correct
        let initial: u32 = py_info.initial;

        // Directly use the values assuming they are already u32
        let finals = py_info.finals.iter().copied().collect::<BTreeSet<u32>>();

        // Transitions conversion assuming all parts are already u32
        let mut transitions = BTreeMap::new();
        for (&(from_state, input), &to_state) in &py_info.transitions {
            transitions.insert((from_state, input), to_state);
        }

        // Alphabet symbol mapping conversion assuming all values are already u32
        let alphabet_symbol_mapping = py_info
            .alphabet_symbol_mapping
            .iter()
            .map(|(symbol, &trans_key)| (symbol.clone(), trans_key))
            .collect::<BTreeMap<String, u32>>();

        // Extract states from transitions
        let mut states = BTreeSet::new();
        for ((from, _), to) in &transitions {
            states.insert(*from);
            states.insert(*to);
        }

        Ok(FSMInfo {
            initial,
            finals,
            transitions,
            alphabet_symbol_mapping,
            states,
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FSMInfo {
    /// Initial state of the FSM
    pub initial: u32,
    /// Final states of the FSM
    pub finals: BTreeSet<u32>,
    /// The transitions map of the fsm.
    /// key is a tuple, (state, input), where the state is the current state, and input is the transition
    /// Value is the state we will end up at if we take the transition
    /// there can be multiple transitions for each state.
    pub transitions: BTreeMap<(u32, u32), u32>,

    /// The alphabet mapping.
    /// key is a String representing the input, value is its u32 identifier / transition key.
    pub alphabet_symbol_mapping: BTreeMap<String, u32>,
    pub states: BTreeSet<u32>,
}

impl FSMInfo {
    /// This method converts a Deterministic Finite Automaton (DFA), provided by the `regex_automata` crate, into a `FSMInfo` structure suitable for use in custom FSM operations.
    ///
    ///- **Exploration of DFA States**: The method explores all states of the DFA to establish reachable states and transitions.
    /// - **Construction of Transition Maps**: It constructs a comprehensive map of transitions and state relationships based on DFA behavior.
    /// - **Pattern Matching**: Utilizes DFA capabilities to identify and classify final states and matchable patterns.
    ///
    /// This conversion facilitates the utilization of complex regular expressions and pattern matching within FSM-based processing systems, providing robust tools for string analysis and manipulation.
    #[cfg(feature = "e2e_experimental")]
    pub fn from_dfa(dfa: &DFA<&[u32]>) -> Self {
        let stride = dfa.stride();
        let stride_u32 = stride as u32;

        let anch = Anchored::Yes;

        let initial = dfa
            .universal_start_state(anch)
            .expect("No Start state for the DFA found. this is un-recoverable.");

        let mut incoming = FxHashMap::default();
        let mut todo = vec![initial];
        incoming.insert(initial, Vec::new());

        // TIME: 1.5ms
        while let Some(s) = todo.pop() {
            for b in 0..=255 {
                let s2 = dfa.next_state(s, b);
                incoming.entry(s2).or_insert_with(|| {
                    todo.push(s2);
                    Vec::new()
                });
                incoming.get_mut(&s2).unwrap().push(s);
            }
        }

        let states = incoming.keys().copied().collect::<Vec<_>>();
        let mut reachable_patterns = FxHashMap::<StateID, Vec<bool>>::default();

        for s in &states {
            let mut patterns = vec![false; dfa.pattern_len()];
            let s2 = dfa.next_eoi_state(*s);
            if dfa.is_match_state(s2) {
                for idx in 0..dfa.match_len(s2) {
                    let idx = dfa.match_pattern(s2, idx).as_usize();
                    patterns[idx] = true;
                }
            }
            reachable_patterns.insert(*s, patterns);
        }

        loop {
            let mut num_set = 0;

            for s in &states {
                let ours = reachable_patterns.get(s).unwrap().clone();
                for o in &incoming[s] {
                    let theirs = reachable_patterns.get(o).unwrap();
                    let mut tmp = ours.clone();
                    tmp.iter_mut()
                        .zip(theirs.iter())
                        .for_each(|(t, &o)| *t |= o);
                    if tmp != *theirs {
                        num_set += 1;
                        reachable_patterns.insert(*o, tmp);
                    }
                }
            }
            if num_set == 0 {
                break;
            }
        }

        let mut states_idx = states.iter().map(|x| x.as_usize()).collect::<Vec<_>>();
        states_idx.sort();

        let transitions = dfa.get_transitions();
        let table = transitions.table();

        let finals: Vec<StateID> = states
            .iter()
            .filter(|&&s| dfa.is_match_state(dfa.next_eoi_state(s)))
            .copied()
            .collect();

        let mut finals_set = BTreeSet::<u32>::new(); // Use BTreeSet here

        for s in finals.iter() {
            finals_set.insert(s.as_u32() / stride_u32);
        }

        let reachable_states = reachable_patterns
            .iter()
            .filter_map(|(&state, reachability)| {
                if reachability.len() == 1 && reachability[0] {
                    Some(state)
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();

        let mut alphabet_symbol_mapping = BTreeMap::<String, u32>::new();
        // Debugging Byte Classes
        let classes = dfa.byte_classes();

        let alphabet_len = dfa.alphabet_len();

        // Iterate over all byte values (0-255)
        for &(start, end) in RANGES.iter() {
            for byte in start..=end {
                let class_index = classes.get(byte) as usize;
                let char = byte as char;
                alphabet_symbol_mapping.insert(char.to_string(), class_index as u32);
            }
        }

        let mut transitions = BTreeMap::new();
        for state in &states {
            if let Some(reachable_vec) = reachable_patterns.get(state) {
                if reachable_vec.get(0).copied().unwrap_or(false) {
                    let i = state.as_usize() / stride;
                    for (j, &state_id) in table
                        .chunks(stride)
                        .nth(i)
                        .unwrap()
                        .iter()
                        .enumerate()
                        .take(alphabet_len)
                    {
                        if state_id.as_u32() > 0 {
                            // no transitioning past the final state. this just brings to the dead state, which causes problems.
                            if !dfa.is_dead_state(state_id) && !dfa.is_match_state(state_id) {
                                transitions.insert(
                                    (state.as_u32() / stride_u32, j as u32),
                                    state_id.as_u32() / stride_u32,
                                );
                            }
                        }
                    }
                }
            }
        }

        FSMInfo {
            initial: initial.as_u32() / stride_u32,
            finals: finals_set,
            transitions: transitions,
            alphabet_symbol_mapping: alphabet_symbol_mapping,
            states: reachable_states
                .iter()
                .map(|x| x.as_u32() / stride_u32)
                .collect::<BTreeSet<_>>(),
        }
    }
}

/// `VocabTrie` is a structure designed to efficiently index the vocabulary of a tokenizer. It facilitates the quick lookup of tokens based on their prefixes and manages relationships between tokens and their substrings, crucial for operations like token scanning in FSMs.
///
/// - **`parent_children_map`**: Maps a token prefix ID to a vector of child token IDs, enabling quick exploration of possible token continuations.
/// - **`idx_to_token_str`**: Provides an index-based lookup from a token ID to the corresponding token string.
/// - **`token_str_to_idx`**: Maps a token string to its unique ID, facilitating fast conversions from strings to indices.
/// VocabTrie is designed for efficient indexing and fast retrieval of tokens,
/// optimized for concurrent read access.
#[derive(Clone)]
pub struct VocabTrie {
    // A sorted vector of tuples (token ID, list of children token IDs) for efficient searching.
    parent_children_map: Arc<Vec<(u32, Vec<u32>)>>,
    // Index-based lookup from token ID to the corresponding token string.
    idx_to_token_str: Arc<Vec<String>>,
    // List of root token indices that have no prefixes, optimized for quick access.
    root_tokens: Arc<Vec<u32>>,
}

impl VocabTrie {
    /// Efficiently finds the children of a given token index using binary search.
    /// Returns an option containing a reference to the children vector if found.
    pub fn find_children(&self, token_idx: u32) -> Option<&[u32]> {
        self.parent_children_map
            .binary_search_by_key(&token_idx, |&(id, _)| id)
            .ok()
            .map(|index| &self.parent_children_map[index].1[..])
    }

    /// Retrieves a token string by its index.
    pub fn get_token(&self, index: u32) -> Option<&String> {
        self.idx_to_token_str.get(index as usize)
    }

    /// Retrieves a reference to the vector of root token indices.
    pub fn get_root_tokens(&self) -> &Vec<u32> {
        &self.root_tokens
    }
}
/// `VocabTrieBuilder` is a trait implemented for `TokenVocabulary` that extends its functionality to include the generation of a `VocabTrie`.
/// This allows any `TokenVocabulary` instance to directly create a trie structure tailored for efficient token handling in FSM operations.
/// Trait for building a VocabTrie from a TokenVocabulary.
pub trait VocabTrieBuilder {
    fn to_vocab_trie(&self) -> VocabTrie;
}

/// Implementation of the VocabTrieBuilder for TokenVocabulary.
impl VocabTrieBuilder for TokenVocabulary {
    fn to_vocab_trie(&self) -> VocabTrie {
        let mut parent_children_map: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut idx_to_token_str: Vec<String> = Vec::new();
        let mut token_str_to_idx: BTreeMap<String, u32> = BTreeMap::new();
        let mut root_tokens: Vec<u32> = Vec::new();

        let mut token_id: u32 = 0;
        for (token, _) in self.iter() {
            idx_to_token_str.push(token.clone());
            token_str_to_idx.insert(token.clone(), token_id);

            // Determine if the token is a root token and manage prefixes
            let char_indices: Vec<usize> = token.char_indices().map(|(index, _)| index).collect();
            let mut is_root = true;
            for i in 0..char_indices.len() - 1 {
                let prefix = &token[..char_indices[i]];
                if self.contains_key(prefix) {
                    let prefix_id = *token_str_to_idx.get(prefix).unwrap();
                    if let Some(entry) = parent_children_map
                        .iter_mut()
                        .find(|entry| entry.0 == prefix_id)
                    {
                        entry.1.push(token_id);
                    } else {
                        parent_children_map.push((prefix_id, vec![token_id]));
                    }
                    is_root = false;
                }
            }
            if is_root {
                root_tokens.push(token_id);
            }
            token_id += 1;
        }

        // Ensure the parent_children_map is sorted by token ID for efficient binary search
        parent_children_map.sort_by_key(|&(id, _)| id);

        VocabTrie {
            parent_children_map: Arc::new(parent_children_map),
            idx_to_token_str: Arc::new(idx_to_token_str),
            root_tokens: Arc::new(root_tokens),
        }
    }
}

#[cfg(feature = "e2e_experimental")]
pub fn build_dfa(pattern: &str, _minimize: bool) -> DFA<Vec<u32>> {
    DFA::builder()
        .configure(DFA::config()
            .start_kind(StartKind::Anchored)
            .match_kind(regex_automata::MatchKind::All))
        .syntax(SyntaxConfig::new()
            .unicode(false)
            .utf8(false))
        .build(pattern)
        .expect("Failed to build DFA. there may be an issue with the pattern, or json schema passed in.")
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//    use std::time::Instant;
//
//    #[test]
//    fn test_fsm_from_dfa_conversion() {
//        // Define a simple pattern that can be used to build the DFA
//        let pattern = r#"\{\s*"location"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"format"\s*:\s*("celsius"|"fahrenheit")\s*\}"#;
//
//        let start_bt = Instant::now();
//        // Build the DFA using the provided build_dfa function
//        let dfa = build_dfa(pattern, true);
//        println!("{:?}", dfa);
//        let end_bt = start_bt.elapsed();
//
//        println!("Time taken to build minimized DFA: {:?}", end_bt);
//
//        let start_bt = Instant::now();
//        // Build the DFA using the provided build_dfa function
//        let dfa = build_dfa(pattern, false);
//        let end_bt = start_bt.elapsed();
//
//        println!("Time taken to build non-minimized DFA: {:?}", end_bt);
//
//        let start = Instant::now();
//       // Convert the DFA to our FSMInfo structure
//        let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());
//        let end = start.elapsed();
//        println!("FSM Info: {:?}", fsm_info);
//       println!("transitions length: {:?}", fsm_info.transitions.len());
//        println!("Time taken: {:?}", end);
//    }
//
//    #[test]
//    fn test_fsm_validity() {
//        // Example pattern for building DFA
//        let pattern = r#"\{\s*"location"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"format"\s*:\s*("celsius"|"fahrenheit")\s*\}"#;
//       // Build the DFA
//        let dfa = build_dfa(pattern, true);
//        // Convert DFA to FSMInfo
//        let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());
//
//       // Start at the initial state
//        let start_state = fsm_info.initial;
//        // Prepare a stack for DFS and a set to track visited states
//        let mut stack = vec![start_state];
//        let mut visited = BTreeSet::new();
//
//        // Depth-first search to explore all reachable states
//        while let Some(state) = stack.pop() {
//            if !visited.insert(state) {
//                continue; // Skip if already visited
//            }
//
//            // Iterate over all possible inputs and push the resulting states to the stack
//            for &input in fsm_info.alphabet_symbol_mapping.values() {
//                if let Some(&next_state) = fsm_info.transitions.get(&(state, input)) {
//                    stack.push(next_state);
//                }
//            }
//       }
//
//        // Now, ensure that all final states are reachable
//        let all_finals_reachable = fsm_info
//            .finals
//            .iter()
//            .all(|final_state| visited.contains(final_state));
//
//        assert!(
//            all_finals_reachable,
//            "Not all final states are reachable from the initial state"
//        );
//    }
//};

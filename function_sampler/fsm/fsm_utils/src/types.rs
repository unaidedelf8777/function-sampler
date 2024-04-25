use regex_automata::{
    dfa::{
        dense::{TransitionTable, DFA},
        Automaton, StartKind,
    },
    util::{primitives::StateID, start::Start},
    Anchored,
};
use std::collections::{BTreeMap, BTreeSet};

pub type TokenVocabulary = BTreeMap<String, Vec<u32>>;

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

#[inline(always)]
fn find_predecessors<'a, T: AsRef<[u32]>>(
    dfa: &'a DFA<T>,
    target_state: &StateID,
    input_symbol: &usize,
    table: &[StateID],
) -> Vec<StateID> {
    dfa.states() // Get pre-multiplied indices // Keep track of original state index
        .filter_map(|start_index| {
            let id = start_index.id();
            let index = id.as_usize() + *input_symbol; // Dereference input_symbol here, only once per invocation
            if &table[index] == target_state {
                Some(id)
            } else {
                None
            }
        })
        .collect()
}

/// filters out terminal control characters, and control char such as escape, backspace, etc.
/// model cant use them so we shouldnt waste time computing for them.
fn is_displayable_char(c: &char) -> bool {
    if c.is_control() {
        // Allow specific control characters: Carriage Return, Tab, Line Feed
        match c {
            '\r' | '\n' | '\t' => true,
            _ => false,
        }
    } else {
        // All other characters that are not control characters are allowed
        true
    }
}

impl FSMInfo {
    pub fn from_dfa(dfa: &DFA<&[u32]>) -> Self {
        let stride = dfa.stride();
        let stride_u32 = stride as u32;
        let initial: StateID = dfa
            .starts()
            .find_map(|(id, anchored, start_type)| {
                if anchored.is_anchored() && start_type == Start::Text {
                    Some(id)
                } else {
                    None
                }
            })
            .expect(
                "PANIC!: No valid initial state found. This should never happen.\n\
        Please open an issue at `https://github.com/unaidedelf8777/function-sampler/issues`, \
        and attach the function or json-spec you used.",
            );

        let initial_idx = initial.as_u32() / stride_u32; // Calculate the initial index.

        let tt: TransitionTable<&[u32]> = dfa.get_transitions();
        let table = tt.table();
        let finals = dfa.get_finals();

        let mut finals_set = BTreeSet::<u32>::new(); // Use BTreeSet here

        let eoi_index = dfa.alphabet_len() - 1;

        for ms in finals.iter() {
            let matches = find_predecessors(dfa, &StateID::from(*ms), &eoi_index, &table);
            for m in matches {
                finals_set.insert(m.as_u32() / stride_u32); // Store the final states as u32 in BTreeSet
            }
        }

        let alphabet_len = tt.alphabet_len();

        let mut alphabet_symbol_mapping = BTreeMap::<String, u32>::new();
        // array of banned transitions. initialized with true.
        let mut banned_trans_indexes = vec![true; alphabet_len];
        // Debugging Byte Classes
        let classes = dfa.byte_classes();

        // Iterate over all byte values (0-255)
        for byte in 0u8..=255 {
            let class_index = classes.get(byte) as usize; // Get the equivalence class index for the byte
            let char = byte as char; // Convert the byte to a char
            if is_displayable_char(&char) {
                alphabet_symbol_mapping.insert(char.to_string(), class_index as u32);
            // Insert the mapping
            } else {
                banned_trans_indexes[class_index] = false; // Mark the transition as banned due to control character presence.
            }
        }

        let special_states_max_id = dfa.get_special_states_max().as_usize() / stride;

        let (accel_max, accel_min) = dfa.get_accel_states_range();
        let (match_max, match_min) = dfa.get_match_states_range();

        let accel_states_min = accel_min.as_usize() / stride;

        let match_states_max = match_max.as_usize() / stride;

        let mut transitions = BTreeMap::new();
        for (i, chunk) in table.chunks(stride).enumerate() {
            if i > special_states_max_id || (i >= accel_states_min && i <= match_states_max) {
                for (j, &state_id) in chunk.iter().enumerate().take(alphabet_len) {
                    if banned_trans_indexes[j] {
                        if !dfa.is_dead_state(state_id) {
                            transitions
                                .insert((i as u32, j as u32), state_id.as_u32() / stride_u32);
                        }
                    }
                }
            }
        }

        println!("num states: {:?}", table.len() / stride);

        FSMInfo {
            initial: initial_idx,
            finals: finals_set,
            transitions: transitions,
            alphabet_symbol_mapping: alphabet_symbol_mapping,
            states: (0..(table.len() / stride)).map(|x| x as u32).collect(),
        }
    }
}

pub fn build_dfa(pattern: &str, minimize: bool) -> DFA<Vec<u32>> {
    DFA::builder()
        .configure(DFA::config()
            .start_kind(StartKind::Anchored)
            .minimize(minimize))
        .build(pattern)
        .expect("Failed to build DFA. there may be an issue with the pattern, or json schema passed in.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_fsm_from_dfa_conversion() {
        // Define a simple pattern that can be used to build the DFA
        let pattern = r#"\{\s*"location"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"format"\s*:\s*("celsius"|"fahrenheit")\s*\}"#;

        let start_bt = Instant::now();
        // Build the DFA using the provided build_dfa function
        let dfa = build_dfa(pattern, true);
        let end_bt = start_bt.elapsed();

        println!("Time taken to build minimized DFA: {:?}", end_bt);

        let start_bt = Instant::now();
        // Build the DFA using the provided build_dfa function
        let dfa = build_dfa(pattern, false);
        let end_bt = start_bt.elapsed();

        println!("Time taken to build non-minimized DFA: {:?}", end_bt);

        let start = Instant::now();
        // Convert the DFA to our FSMInfo structure
        let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());
        let end = start.elapsed();
        println!("{:?}", fsm_info);
        println!("transitions length: {:?}", fsm_info.transitions.len());
        println!("Time taken: {:?}", end);
    }

    #[test]
    fn test_fsm_validity() {
        // Example pattern for building DFA
        let pattern = r#"\{\s*"location"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"format"\s*:\s*("celsius"|"fahrenheit")\s*\}"#;
        // Build the DFA
        let dfa = build_dfa(pattern, true);
        // Convert DFA to FSMInfo
        let fsm_info = FSMInfo::from_dfa(&dfa.as_ref());

        // Start at the initial state
        let start_state = fsm_info.initial;
        // Prepare a stack for DFS and a set to track visited states
        let mut stack = vec![start_state];
        let mut visited = BTreeSet::new();

        // Depth-first search to explore all reachable states
        while let Some(state) = stack.pop() {
            if !visited.insert(state) {
                continue; // Skip if already visited
            }

            // Iterate over all possible inputs and push the resulting states to the stack
            for &input in fsm_info.alphabet_symbol_mapping.values() {
                if let Some(&next_state) = fsm_info.transitions.get(&(state, input)) {
                    stack.push(next_state);
                }
            }
        }

        // Now, ensure that all final states are reachable
        let all_finals_reachable = fsm_info
            .finals
            .iter()
            .all(|final_state| visited.contains(final_state));

        assert!(
            all_finals_reachable,
            "Not all final states are reachable from the initial state"
        );
    }
}

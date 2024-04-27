use regex_automata::{
    dfa::{dense::DFA, Automaton, StartKind},
    util::{primitives::StateID, syntax::Config as SyntaxConfig},
    Anchored,
};
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};

pub type TokenVocabulary = BTreeMap<String, Vec<u32>>;

lazy_static! {
    // very tiny optim so we dont need to iterate over unnesescarry bytes we will never use.
    static ref RANGES: [(u8, u8); 8] = [
        (97, 122),   // Lowercase letters [a-z]
        (65, 90),    // Uppercase letters [A-Z]
        (48, 57),    // Digits [0-9]
        (32, 47),    // Punctuation and symbols (first part)
        (58, 64),    // Punctuation and symbols (second part)
        (91, 96),    // Punctuation and symbols (third part)
        (123, 126),  // Punctuation and symbols (fourth part)
        (9, 13),     // Whitespace and control characters (tab, newline, carriage return)
    ];
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

pub fn build_dfa(pattern: &str, _minimize: bool) -> DFA<Vec<u32>> {
    DFA::builder()
        .configure(DFA::config()
            .start_kind(StartKind::Anchored))
        .syntax(SyntaxConfig::new()
            .unicode(false)
            .utf8(false))
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
        let pattern = r#"\{[\n ]*"restaurant_name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,([\n ]*"reservation_date"[\n ]*:[\n ]*(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])[\n ]*,)?[\n ]*"reservation_time"[\n ]*:[\n ]*(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?([\n ]*,[\n ]*"party_size"[\n ]*:[\n ]*(-)?(0|[1-9][0-9]*))?([\n ]*,[\n ]*"contact_number"[\n ]*:[\n ]*(-)?(0|[1-9][0-9]*))?[\n ]*\}"#;

        let start_bt = Instant::now();
        // Build the DFA using the provided build_dfa function
        let dfa = build_dfa(pattern, true);
        println!("{:?}", dfa);
        let end_bt = start_bt.elapsed();

        println!(
            "DFA STATE 1: {:?}",
            dfa.states()
                .nth(60)
                .map(|state| state.transitions().collect::<Vec<_>>())
        );

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
        println!("FSM Info: {:?}", fsm_info);
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

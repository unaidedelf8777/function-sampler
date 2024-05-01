
use pyo3::prelude::*;

use std::collections::BTreeMap;

use std::sync::{Arc, Condvar, Mutex};

use std::thread;

use crate::{
    tokenizer_index::create_fsm_index_end_to_end_parallel,
    types::{FSMInfo, TokenVocabulary},
};

#[pyclass]
#[derive(Clone, Debug)]
pub struct LazyFSMIndex {
    /// *Public fields* ///
    /// ( public meaning they are accesable either in full or in part from python ) ///

    /// the mapping of states to token subsets from the tokenizer.
    /// this is an interpreted version of the FSM info.
    /// interpreted according to the token vocabulary.
    states_to_token_maps: Arc<Mutex<BTreeMap<u32, BTreeMap<u32, u32>>>>,

    /// First state of the FSM
    first_state: u32,

    /// The end-of-sequence token ID from tokenizer.
    eos_token_id: u32,

    /// *Private fields* ///
    /// ( not accesable from python )

    /// struct holding all info of the fsm.
    /// struct details and construction located in
    /// crate::types::FSMInfo
    fsm_info: Arc<FSMInfo>,

    // for notifying waiters when a state is finished.
    state_notifiers: Arc<Mutex<BTreeMap<u32, Arc<(Mutex<bool>, Condvar)>>>>,

    /// bool indicator, just so we dont need to manually iterate
    /// over the notifiers to check if they are all finished.
    computing_finished: Arc<Mutex<bool>>,
}

/// We gate the creation of new indexes in a impl block seperate to that of the definition
/// of all of the python methods, since the only way to construct a new LazyFSMIndex is through
/// `create_fsm_index_end_to_end_py` function interfaced to python.

impl LazyFSMIndex {
    pub fn new(fsm_info: FSMInfo, vocabulary: TokenVocabulary, eos_token_id: u32) -> Self {
        let fsm_info_arc = Arc::new(fsm_info);
        let fsm_info_arc_clone = Arc::clone(&fsm_info_arc);
        let vocabulary_arc = Arc::new(vocabulary);
        let results = Arc::new(Mutex::new(BTreeMap::<u32, BTreeMap<u32, u32>>::new()));
        let state_notifiers = Arc::new(Mutex::new(
            BTreeMap::<u32, Arc<(Mutex<bool>, Condvar)>>::new(),
        ));
        let state_notifiers_clone = Arc::clone(&state_notifiers);

        let computing_finished = Arc::new(Mutex::new(false));
        let computing_finished_clone = Arc::clone(&computing_finished);

        // Clone the results so we can pass it to the thread.
        // this way we keep ownership of results, but the thread can still update the results var we have.
        let results_clone = Arc::clone(&results);

        // Start the computation in a new thread
        thread::spawn(move || {
            create_fsm_index_end_to_end_parallel(
                &fsm_info_arc,
                &vocabulary_arc,
                &results_clone,
                &state_notifiers_clone,
            );
            *computing_finished_clone.lock().unwrap() = true;
        });

        let first_state = fsm_info_arc_clone.initial;

        return LazyFSMIndex {
            states_to_token_maps: results,
            eos_token_id,
            fsm_info: fsm_info_arc_clone,
            computing_finished,
            first_state: first_state,
            state_notifiers,
        };
    }
}

/// implementation of all the python methods for the LazyFSMIndex struct.

#[pymethods]
impl LazyFSMIndex {
    /// Retrieves the state map for a given state, waiting if it's not immediately available.
    fn get_state_map(&self, state: u32) -> Option<BTreeMap<u32, u32>> {
        {
            // First attempt to get the state map without waiting
            let state_map_lock = self.states_to_token_maps.lock().unwrap();
            if let Some(token_map) = state_map_lock.get(&state) {
                return Some(token_map.clone());
            }
        } // Explicitly drop the lock

        // If the state is not found, find or create a notifier and wait
        let notifier = {
            let mut notifiers = self.state_notifiers.lock().unwrap();
            notifiers
                .entry(state)
                .or_insert_with(|| Arc::new((Mutex::new(false), Condvar::new())))
                .clone()
        };

        // Access the tuple inside the Arc correctly
        let done_lock = &Arc::clone(&notifier).0;
        let cvar = &Arc::clone(&notifier).1;
        let mut done = done_lock.lock().unwrap();
        while !*done {
            done = cvar.wait(done).unwrap();
        }

        // Try again to fetch the state map after being notified
        let state_map_lock = self.states_to_token_maps.lock().unwrap();
        state_map_lock.get(&state).cloned()
    }

    pub fn next_state(&self, state: i32, token_id: u32) -> Option<i32> {

        // check if they are alias states first ( -1, or 0 )
        // state 0 is alias for the first state
        // -1 alias for the last state.
        // if the state is already final, then the next state can only ever be final.
        if state == -1 {
            return Some(-1);
        }



        // Check if the token ID is the end-of-sequence or the state is a final state
        if token_id == self.eos_token_id || self.fsm_info.finals.contains(&(state as u32)) {
            return Some(-1);
        }

        let current_state = if state == 0 { self.first_state } else { state as u32 };


        // Attempt to find the next state using the get_state_map method
        let token_map = self.get_state_map(current_state);
        token_map
            .and_then(|map| map.get(&token_id).copied().map(|s| s as i32))
            // if the token to next state pair is not found, return -1, this isnt something we can blindly repair.
            // may as well just give up!
            .or(Some(-1))
    }

    pub fn is_final_state(&self, state: i32) -> bool {
        // py version:
        // return state == self.final_state

        // Check if the state is the "final" or invalid state
        if state == -1 || self.fsm_info.finals.contains(&(state as u32)) {
            return true;
        } else {
            return false;
        }
    }

    pub fn is_computing_finished(&self) -> bool {
        *self.computing_finished.lock().unwrap()
    }

    pub fn get_states_to_token_subsets(&self) -> BTreeMap<u32, BTreeMap<u32, u32>> {
        // we first need to wait for the computation to finish.
        // after computation is finished, then we can return it.
        // this will block the thread until we can return the full result.
        while !self.is_computing_finished() {
            // Sleep for a short time to avoid busy waiting
            thread::sleep(std::time::Duration::from_millis(2));
        }
        // finished. return a clone.
        self.states_to_token_maps.lock().unwrap().clone()
    }

    pub fn allowed_token_ids(&self, state: i32) -> Vec<i32> {
        if state == -1 {
            return vec![-2];
        }
        match self.get_state_map(state as u32) {
            Some(next_tokens_to_end_states) => {
                // Collect all keys (token IDs) from the map and convert them to i32
                next_tokens_to_end_states
                    .keys()
                    .cloned()
                    .map(|k| k as i32)
                    .collect()
            }
            None => {
                // if allowed tokens is [-2] means
                // no tokens are allowed.
                vec![-2]
            }
        }
    }
}

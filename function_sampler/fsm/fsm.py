from typing import TYPE_CHECKING, List, NewType, Protocol, Tuple

import interegular

from .regex import create_fsm_index_tokenizer, make_deterministic_fsm
import time

if TYPE_CHECKING:
    from .tokenizer_fsm_patch import Tokenizer

FSMState = NewType("FSMState", int)


class FSM(Protocol):

    def is_final_state(self, state: FSMState) -> bool:
        """Determine whether the current state of the FSM is a final state."""
        return state in self.final_state

    def allowed_token_ids(self, state: FSMState) -> List[int]: ...

    def next_state(self, state: FSMState, token_id: int) -> FSMState: ...



class RegexFSM(FSM):
    """FSM to generate text that is in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer):
        def create_states_mapping(
            regex_string: str, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            start_time = time.perf_counter()
            states_to_token_maps, empty_token_ids, initial, finals = create_fsm_index_tokenizer(
                regex_string, tokenizer
            )

            print(len(states_to_token_maps))

            # cant handle this right now, rust doesnt support it, 
            # I'm pretty sure rust panics if this err happens tho
            #
            # vocabulary.
            # if not any(
            #     regex_fsm.finals.intersection(v.values())
            #     for v in states_to_token_maps.values()
            # ):
            #     raise ValueError(
            #         "The vocabulary does not allow us to build a sequence that matches the input regex"
            #    )

            return states_to_token_maps, empty_token_ids, initial, finals

        self.states_to_token_maps, self.empty_token_ids, self.first_state, self.final_states = create_states_mapping(
            regex_string, tuple(sorted(tokenizer.vocabulary.items()))
        )
        self.vocabulary = list(tokenizer.vocabulary.values())
        self.eos_token_id = tokenizer.eos_token_id

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        """Generate a list of allowed tokens for the next step.

        The initialization of the FSM builds an index which maps FSM states to a
        map from authorized tokens to the state in which the FSM needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the FSM. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
        next_tokens_to_end_states = self.states_to_token_maps.get(state)

        if next_tokens_to_end_states is None:
            return [-2]
        else:
            return list(next_tokens_to_end_states.keys())

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        """Update the state of the FSM.

        We use the index to determine to which state the FSM should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.

        """
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            return self.final_state

        return FSMState(next_state)

    def copy(self) -> "RegexFSM":
        """Create a copy of the FSM."""
        return self

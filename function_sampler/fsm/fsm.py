from typing import TYPE_CHECKING, List, NewType, Protocol, Tuple

import interegular

from .regex import create_fsm_index_tokenizer, make_deterministic_fsm
import time

if TYPE_CHECKING:
    from .tokenizer_fsm_patch import Tokenizer

FSMState = NewType("FSMState", int)


class FSM(Protocol):
    first_state: FSMState = FSMState(0)
    final_state: FSMState = FSMState(-1)

    def is_final_state(self, state: FSMState) -> bool:
        """Determine whether the current state of the FSM is a final state."""
        return state == self.final_state

    def allowed_token_ids(self, state: FSMState) -> List[int]: ...

    def next_state(self, state: FSMState, token_id: int) -> FSMState: ...

    def copy(self) -> "FSM": ...


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
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
                regex_fsm, tokenizer
            )

            # We make sure that it is possible to generate strings in the language
            # of the regular expression with the tokens present in the model's
            # vocabulary.
            if not any(
                regex_fsm.finals.intersection(v.values())
                for v in states_to_token_maps.values()
            ):
                raise ValueError(
                    "The vocabulary does not allow us to build a sequence that matches the input regex"
                )
            end_time = time.perf_counter()
            print(f"Time taken for Rust: {end_time - start_time} seconds")
            return states_to_token_maps, empty_token_ids

        self.states_to_token_maps, self.empty_token_ids = create_states_mapping(
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

    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer: "Tokenizer"
    ):
        from_interegular_instance = cls.__new__(cls)

        def create_states_mapping_from_interegular_fsm(
            fsm: interegular.fsm.FSM, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            regex_fsm, _ = make_deterministic_fsm(fsm.reduce())
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
                regex_fsm, tokenizer
            )

            # We make sure that it is possible to generate strings in the language
            # of the regular expression with the tokens present in the model's
            # vocabulary.
            if not any(
                regex_fsm.finals.intersection(v.values())
                for v in states_to_token_maps.values()
            ):
                raise ValueError(
                    "The vocabulary does not allow us to build a sequence that matches the input regex"
                )

            return states_to_token_maps, empty_token_ids

        (
            from_interegular_instance.states_to_token_maps,
            from_interegular_instance.empty_token_ids,
        ) = create_states_mapping_from_interegular_fsm(
            interegular_fsm, tuple(sorted(tokenizer.vocabulary.items()))
        )
        from_interegular_instance.vocabulary = list(tokenizer.vocabulary.values())
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        return from_interegular_instance

    def copy(self) -> "RegexFSM":
        """Create a copy of the FSM."""
        return self

from typing import TYPE_CHECKING, Dict, Set, Tuple, NewType

from function_sampler.fsm import create_fsm_index_end_to_end
from .utils import reduced_vocabulary

if TYPE_CHECKING:
    from .tokenizer_fsm_patch import Tokenizer

FSMState = NewType("FSMState", int)


def create_fsm_index_tokenizer(
    pattern: str,
    tokenizer: "Tokenizer",
) -> Tuple[Dict[int, Dict[int, int]], Set[int]]:
    """Construct an FMS index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.

    .. warning::

        `fsm` needs to be deterministically ordered so that future caching makes sense.

    """
    vocabulary, empty_token_ids = reduced_vocabulary(tokenizer)

    # rust impl expects generic types, so just cast them.
    states_to_token_subsets = create_fsm_index_end_to_end(
        pattern, dict(vocabulary), tokenizer.eos_token_id
    )  # type: ignore
    return states_to_token_subsets, empty_token_ids

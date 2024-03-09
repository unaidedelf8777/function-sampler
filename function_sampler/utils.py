import torch
from .logger import get_logger
from .fsm import RegexFSM, FsmTokenizer
from .json import build_regex_from_schema
from json import dumps as json_dumps
from transformers import PreTrainedTokenizer

logger = get_logger()


def build_masks(tokenizer, vocab_size, token_masks, json_tokens):
    bad_tokens = []
    if tokenizer.eos_token_id:
        bad_tokens.append(tokenizer.eos_token_id)
    if tokenizer.bos_token_id:
        bad_tokens.append(tokenizer.bos_token_id)
    if tokenizer.unk_token_id:
        bad_tokens.append(tokenizer.unk_token_id)

    for key, token_indexes in json_tokens.items():
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for index in token_indexes:
            if index not in bad_tokens:
                mask[index] = True
        token_masks[key] = mask


def tokenize_dicts(
    input_dicts,
    tokenizer,
    exempt_keys=["required", "name", "type", "enum", "parameters"],
    root=True,
):
    """
    Tokenize keys of dictionaries using a specified tokenizer,
    excluding keys listed in exempt_keys. Specifically handles 'required' list elements.

    :param input_dicts: A list of dictionaries to process.
    :param exempt_keys: A list of keys to exclude from tokenization.
    :return: A new dictionary with tokenized keys where applicable.
    """

    tokenized_dicts = {}
    for input_dict in input_dicts:
        # Tokenizing the 'name' key for the main dictionary key
        name_str = '"' + input_dict["name"] + '",'
        name_str = name_str.strip()
        # Tokenizers are wierd. we need to encode it for its final representation ( after the model "generates" it ),
        # otherwise decoding won't work because of the causal encoding ( tokens can only see tokens before them ).
        # So if a single quote isn't a thing for you, you're out of luck, and the entire lib will explode :)
        name_tokens = tokenizer.encode(name_str, add_special_tokens=False)[1:]
        logger.debug(tokenizer.decode(name_tokens))
        name_tuple = tuple(name_tokens)
        tokenized_dicts[name_tuple] = input_dict["parameters"]

    return tokenized_dicts


def compute_fsm(tokenizer: FsmTokenizer, schema):
    if isinstance(tokenizer, PreTrainedTokenizer):
        tokenizer = FsmTokenizer(tokenizer)

    regex = build_regex_from_schema(json_dumps(schema))
    fsm = RegexFSM(regex, tokenizer)
    return fsm


def temperature_sample(scores: torch.FloatTensor, temperature) -> torch.FloatTensor:
    scores = scores / temperature
    return scores


def sample_top_p(
    scores: torch.FloatTensor,
    top_p: float,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    top_p = float(top_p)
    if top_p < 0 or top_p > 1.0:
        raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def sample_top_k(
    scores: torch.FloatTensor,
    top_k: float,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    top_k = min(top_k, scores.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def sample_repetition_penalty(
    input_ids: torch.LongTensor, scores: torch.FloatTensor, penalty: float
) -> torch.FloatTensor:
    if not isinstance(penalty, float) or not (penalty > 0):
        raise ValueError(
            f"`penalty` has to be a strictly positive float, but is {penalty}"
        )

    score = torch.gather(scores, 1, input_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores.scatter_(1, input_ids, score)
    return scores


def bundle_sampling(
    scores: torch.FloatTensor,
    input_ids: torch.LongTensor = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    repetition_penalty: float = None,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    # Apply temperature scaling if defined
    if temperature is not None:
        scores = scores / temperature

    # Apply top-p sampling if defined
    if top_p is not None:
        scores = sample_top_p(scores, top_p, filter_value, min_tokens_to_keep)

    # Apply top-k sampling if defined
    if top_k is not None:
        scores = sample_top_k(scores, top_k, filter_value, min_tokens_to_keep)

    # Apply repetition penalty if defined and input_ids are provided
    if repetition_penalty is not None and input_ids is not None:
        scores = sample_repetition_penalty(input_ids, scores, repetition_penalty)

    return scores

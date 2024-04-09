import time
import function_sampler
from function_sampler.fsm import FsmTokenizer
from transformers import AutoTokenizer
from function_sampler.json import build_regex_from_schema
from function_sampler.cache import clear_cache, disable_cache
from json import dumps as json_dumps

disable_cache()
clear_cache()
s = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    },
    {
        "name": "get_reservation",
        "description": "Retrieve a reservation at a restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant_name": {
                    "type": "string",
                    "description": "The name of the restaurant for which the reservation is made",
                },
                "reservation_date": {
                    "type": "string",
                    "format": "date",
                    "description": "The date of the reservation in YYYY-MM-DD format",
                },
                "reservation_time": {
                    "type": "string",
                    "format": "time",
                    "description": "The time of the reservation in HH:MM format",
                },
                "party_size": {
                    "type": "integer",
                    "description": "The number of people included in the reservation",
                },
                "contact_number": {
                    "type": "integer",
                    "description": "The contact phone number for the reservation confirmation",
                },
            },
            "required": ["restaurant_name", "reservation_time"],
        },
    },
]


function_sampler.cache.disable_cache()


# A function to perform the benchmark test
def test_benchmark_compile_fsm():
    """Benchmark the numba compilation time without mocker."""

    # Reload the module to apply the patched njit
    from function_sampler.fsm import RegexFSM

    pattern1 = build_regex_from_schema(json_dumps(s[0]["parameters"]))
    pattern2 = build_regex_from_schema(json_dumps(s[1]["parameters"]))
    tokenizer = FsmTokenizer(
        AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    )

    # Benchmark phase

    for i in range(2):
        if i == 0:
            start_time_rs = time.perf_counter()
            fsm = RegexFSM(pattern1, tokenizer)
            end_time_rs = time.perf_counter()
            print(fsm)
            print(f"Time taken for Rust: {end_time_rs - start_time_rs} seconds")
            clear_cache()
            disable_cache()
            print("====================================")
            print(
                f"initial tokens: {[tokenizer.decode([x])[0] for x in fsm.allowed_token_ids(fsm.first_state)]}"
            )
            print("====================================")
        elif i == 1:
            start_time_rs = time.perf_counter()
            fsm = RegexFSM(pattern2, tokenizer)
            end_time_rs = time.perf_counter()
            print(fsm)
            print(f"Time taken for Rust: {end_time_rs - start_time_rs} seconds")
            clear_cache()
            disable_cache()
            print("====================================")
            print(
                f"initial tokens: {[tokenizer.decode([x])[0] for x in fsm.allowed_token_ids(fsm.first_state)]}"
            )
            print("====================================")


# Run the benchmark test
test_benchmark_compile_fsm()

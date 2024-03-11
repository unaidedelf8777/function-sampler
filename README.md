# Function Sampler

<a target="_blank" href="https://colab.research.google.com/github/unaidedelf8777/function-sampler/blob/main/notebooks/Tool_Call_Sampler_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Function Sampler is a powerful library that provides a novel approach to enforcing structured generation on language models. Unlike other libraries such as Langchain or Llama Index, which rely on prompts and hope that the model follows the prompt for parseable outputs, Function Sampler makes it probabilistically impossible for the language model to output invalid function calls.

By using Logit sampling and a Finite State Machine (FSM), Function Sampler guides the language model to generate function calls that adhere to a predefined schema. This eliminates the need for parsing the outputs and ensures that the generated function calls are always valid.

## Features

- Enforces the schema of function calls on the language model using Logit sampling
- Activates sampling based on a specified delimiter token or string in the configuration
- Supports top_p, top_k, temperature, and repetition_penalty sampling for function call values
- Utilizes a Finite State Machine (FSM) to guide the sampling process
- Provides a flexible configuration system using Pydantic models or keyword arguments
- Includes a demo notebook showcasing various usage examples

## Installation

To install the `function-sampler` library, use the following command:

```bash
pip install function-sampler
```

## Usage

Here's a basic example of how to use the `function-sampler` library:

```python
from function_sampler import ToolCallSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the tokenizer and model
# if using a small GPU, or low vram:
# tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")

# Define the functions
functions = [
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
    }
]

# Configure the sampler
config = {
    "open_func_token": "<function>",
    "close_func_token": "</function>",
    "end_on_function_call": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Create an instance of ToolCallSampler
sampler = ToolCallSampler(tokenizer, functions, config=config)


# Use the model for generation
# only need to tell it how to call the function if it is not explicitly trained for it.
input_text = "What is the weather today in paris? respond with the word '<function>' to call the weather API."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=200, logits_processor=[sampler])
generated_text = tokenizer.decode(output[0])
print(generated_text)
# <function>  {"name": "get_reservation", "arguments": {"restaurant_name": "Maggiano's", "reservation_time": 18:00:00, "party_size": 6, "contact_number": 1234567890}} </function><|im_end|>
```

In this example, we create an instance of the `ToolCallSampler` with the specified functions and configuration. We then attach the sampler to the model's `logits_processor` attribute. This ensures that the sampler is applied during the generation process.

Finally, we use the model to generate text based on the input prompt, which includes the opening function token. The generated text will contain a valid function call adhering to the predefined schema.

For more detailed usage and examples, please refer to the demo notebook provided with the library.

## Configuration

The `function-sampler` library offers a flexible configuration system. You can customize the behavior of the sampler by providing a configuration dictionary, a `ToolCallSamplerConfig` instance, or keyword arguments when initializing the `ToolCallSampler` class.

The available configuration options include:

- `open_func_token`: The opening delimiter token for a function call (default: `"<function>"`)
- `close_func_token`: The closing delimiter token for a function call (default: `"</function>"`)
- `end_on_function_call`: Whether to end the generation when a function call is encountered (default: `False`)
- `json_tokens`: A custom token map for JSON tokens (default: built from the provided tokenizer)
- `temperature`: The temperature value for sampling (default: `None`)
- `top_p`: The top_p value for sampling (default: `None`)
- `top_k`: The top_k value for sampling (default: `None`)
- `repetition_penalty`: The repetition penalty value for sampling (default: `None`)

## Contributing

Contributions to the `function-sampler` library are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

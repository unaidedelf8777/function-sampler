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

Before installing the `function-sampler` library, you need to first ensure that the Rust programming language is installed on your system. Follow the installation instructions for your platform below, then continue to install the library from source.

### Install Rust

<details>
<summary><strong>Windows</strong></summary>

1. Download and run the Rust installer from [rustup.rs](https://rustup.rs/).
2. Follow the prompts to install Rust. This will also install `cargo`, Rust's package manager and build system.
3. After installation, open a new command prompt and verify the installation by running:

    ```bash
    rustc --version
    ```

4. Add Rust to your system PATH manually if it's not done automatically by the installer. Usually, Rust is installed under `%USERPROFILE%\.cargo\bin`.
5. If Rust is installed correctly, you should see the version number, commit hash, and commit date.
</details>

<details>
<summary><strong>macOS</strong></summary>

1. You can install Rust using the following command in your terminal:

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2. Follow the instructions on the screen to complete the installation.
3. After the installation is complete, restart your terminal and verify the installation by running:

    ```bash
    rustc --version
    ```

4. Rust installs its binaries in `~/.cargo/bin`. You may need to add this directory to your PATH using:

    ```bash
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bash_profile
    ```

5. If Rust is installed correctly, you should see the version number, commit hash, and commit date.
</details>

<details>
<summary><strong>Linux</strong></summary>

1. Use the following command in your terminal to install Rust:

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2. Follow the on-screen instructions to complete the installation process.
3. After completing the installation, source the Rust environment script:

    ```bash
    source $HOME/.cargo/env
    ```

5. Verify the installation by running:

    ```bash
    rustc --version
    ```

6. If Rust is installed correctly, you should see the version number, commit hash, and commit date.
</details>

### Install `function-sampler` from Source

*Note*: Currently, until I can get the CI for PyPI sorted out, install from source is necessary.

```bash
git clone https://github.com/unaidedelf8777/function-sampler.git
cd function-sampler
python setup.py install
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

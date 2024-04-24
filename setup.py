from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension
import toml

# Load the contents of your pyproject.toml
with open('pyproject.toml', 'r') as pyproject_file:
    pyproject_data = toml.load(pyproject_file)

# Extract package metadata
metadata = pyproject_data.get('tool', {}).get('poetry', {})

# Define the Rust extension module, assuming it's the same name as the package name
rust_extensions = [
    RustExtension(
        f"function_sampler.fsm.fsm_utils",  # Python package.module for the Rust extension
        f"./Cargo.toml",  # Path to Cargo.toml of Rust extension
        binding=Binding.PyO3
    ),
]

setup(
    name=metadata.get('name'),
    version=metadata.get('version'),
    author=",".join(metadata.get('authors', [])),
    author_email=metadata.get('emails', [''])[0],
    description=metadata.get('description', ''),
    long_description=metadata.get('description', ''),
    long_description_content_type="text/markdown",
    url=metadata.get('homepage', ''),
    classifiers=metadata.get('classifiers', []),
    keywords=",".join(metadata.get('keywords', [])),
    license=metadata.get('license', ''),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9',
    setup_requires=['setuptools-rust', 'wheel', 'toml'],
    rust_extensions=rust_extensions,
)


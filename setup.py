from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


metadata = {
    'name': "function-sampler",
    'version': "0.1.5",
    'description': "Function calling Logit Sampler",
    'long_description': "", 
    'authors': ["unaidedelf8777"],
    'author_email': "thwackyy.y@gmail.com",
    'license': "Apache 2.0",
    'classifiers': [
        "Programming Language :: Rust",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    'url': "https://github.com/unaidedelf8777/function-sampler/",
}

rust_extensions = [
    RustExtension(
        f"function_sampler.fsm.fsm_utils",
        f"./Cargo.toml",
        binding=Binding.PyO3,
        features=["default"],
        args=["--profile=release"]
    ),
]

setup(
    name=metadata['name'],
    version=metadata['version'],
    author=metadata['authors'][0].split(" <")[0],  # Extracting name
    author_email=metadata['author_email'],
    description=metadata['description'],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=metadata['url'],
    classifiers=metadata['classifiers'],
    license=metadata['license'],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9',
    setup_requires=['setuptools_rust', 'wheel'],
    rust_extensions=rust_extensions,
)

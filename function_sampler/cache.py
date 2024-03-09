from joblib import Memory
import os

# Determine the cache directory, defaulting to ~/.cache/function_sampler
home_dir = os.path.expanduser("~")
cache_dir = os.environ.get(
    "FUNCTION_SAMPLER_CACHE_DIR", f"{home_dir}/.cache/function_sampler"
)

# Create a Memory object for caching to the specified directory
memory = Memory(cache_dir, verbose=0)


def cache(cached_function):
    """Caching decorator for memoizing function calls using joblib."""
    cached_func = memory.cache(cached_function)

    def wrapper(*args, **kwargs):
        # joblib's cache mechanism automatically handles args and kwargs
        return cached_func(*args, **kwargs)

    # If the function is asynchronous, you'll need additional handling
    # since joblib does not directly support async functions.
    return wrapper


_caching_enabled = True


def disable_cache():
    """Disable the cache for this session."""
    global _caching_enabled
    _caching_enabled = False


def clear_cache():
    """Erase the cache completely."""
    memory.clear(
        warn=False
    )  # warn=False suppresses warnings if the cache directory is already empty

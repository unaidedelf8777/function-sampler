import logging
from .sampler import ToolCallSampler
from .config.config import ToolCallSamplerConfig

__all__ = ["ToolCallSampler", "ToolCallSamplerConfig"]

__version__ = "0.1.0"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("function-sampler")
console_handler = logging.StreamHandler()

logger.addHandler(console_handler)

import logging


def get_logger(name="function-sampler"):
    """
    Returns the singleton logger instance with the given name.

    Args:
        name (str): Name of the logger to retrieve. Defaults to 'appLogger'.

    Returns:
        logging.Logger: The configured logger instance.
    """
    return logging.getLogger(name)

import logging
import time
from functools import wraps
from contextlib import contextmanager


def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer(logger, operation):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{operation} took {elapsed:.2f} seconds")


def log_execution_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} took {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator

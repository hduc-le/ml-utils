import logging
import time as _time
from datetime import datetime
from functools import wraps as _wraps

logger = logging.getLogger(__name__)


def logger_func(call_depth=0):
    def decorator(function):
        @_wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal call_depth
            indent = " " * call_depth
            logger.info(
                f"{indent}----- {function.__name__}: start AT {datetime.now()}-----"
            )
            start = _time.perf_counter()
            call_depth += 1
            try:
                output = function(*args, **kwargs)
            finally:
                call_depth -= 1
            end = _time.perf_counter()
            logger.info(
                f"{indent}----- {function.__name__}: end took {end - start:.6f} | {datetime.now()} seconds to complete -----"
            )
            return output

        return wrapper

    return decorator

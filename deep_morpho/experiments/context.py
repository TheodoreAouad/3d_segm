from typing import Any

import logging
from time import time

from general.utils import format_time, log_console


class Task:
    """Context manager logging when a task is done."""

    def __init__(self, message: str, console_logger: logging.Logger = None, verbose: bool = True):
        self.message = message
        self.console_logger = console_logger
        self.verbose = verbose

    def __enter__(self) -> None:
        self.start_time = time()
        self.log_console(self.message)

    def __exit__(self, *args: Any) -> None:
        self.log_console(f"Done in {format_time(time() - self.start_time)}")

    def log_console(self, message):
        if self.verbose:
            log_console(message, logger=self.console_logger)

#!/usr/bin/env python

import time


class Timer:
    """Context manager to measure time of things inside it"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.name}: {elapsed} seconds")

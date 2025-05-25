from utils import Timer
from io import StringIO
import sys
import time


def test_timer_context_manager():
    # Redirect stdout to capture the print output from Timer
    captured_output = StringIO()
    sys.stdout = captured_output

    with Timer("test_timer"):
        time.sleep(0.05)

    sys.stdout = sys.__stdout__  # Reset stdout to its original state

    # Extracting the printed time measurement
    output = captured_output.getvalue()
    assert output.startswith("test_timer: 0.05")


test_timer_context_manager()

import os


if __package__ is None or __package__ == "":
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    sys.path.insert(0, os.path.dirname(os.path.dirname(script_dir)))
    __package__ = "model.exp17"

from model.eval_runners import run_eval


if __name__ == "__main__":
    run_eval(__package__, include_voltage_linear_prior=True)

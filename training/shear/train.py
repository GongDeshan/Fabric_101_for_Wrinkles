"""Launch the shear model with the original command-line and GUI behavior.

Examples:
    python train.py --headless-train --fabric 59_Cotton --direction warp --epochs 10
    python train.py

The full editable simulation and optimization script is shear_model.py.
"""
from pathlib import Path
import runpy


def main():
    runpy.run_path(str(Path(__file__).resolve().with_name("shear_model.py")),
                   run_name="__main__")


if __name__ == "__main__":
    main()

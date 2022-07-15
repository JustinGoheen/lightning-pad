import platform

import lightning_app
import mypy
import numpy as np
import pre_commit
import pytest
import pytorch_lightning
import sklearn
import torch
import torchmetrics
from rich import print as rprint

libs = [
    torch,
    pytorch_lightning,
    lightning_app,
    torchmetrics,
    np,
    sklearn,
    pytest,
    mypy,
    pre_commit,
]

print()
rprint("[bold green]ENV VERSIONS:[/bold green]", end="\n\n")
print("python" + ": ", platform.python_version())
for lib in libs:
    if hasattr(lib, "__version__"):
        print(
            lib.__name__ + ": ",
            lib.__version__,
            "\n" if lib.__name__ == libs[-1].__name__ else "",
        )
    else:
        print(
            lib.__name__ + ": ",
            "installed",
            "\n" if lib.__name__ == libs[-1].__name__ else "",
        )

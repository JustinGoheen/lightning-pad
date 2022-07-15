import platform

import lightning_app as la
import mypy
import numpy as np
import pre_commit
import pytest
import pytorch_lightning as pl
import sklearn as sk
import torch as t
import torchmetrics as tm
from rich import print as rprint

libs = [t, pl, la, tm, np, sk, pytest, mypy, pre_commit]

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

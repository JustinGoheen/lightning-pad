import numpy as np
import torchmetrics as tm
import sklearn as sk
import torch as t
import pytorch_lightning as pl
import lightning_app as la
from rich import print as rprint
import platform
import pytest
import mypy
import pre_commit

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

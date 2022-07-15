import numpy as np
import pandas as pd
import torchmetrics as tm
import sklearn as sk
import torch as t
import pytorch_lightning as pl
from rich import print as rprint
import sys


rprint("[bold green]collecting env installs[/bold green]", end="\n\n")
print("python", sys.version)
for lib in [np, pd, tm, sk, t, pl]:
    print(
        lib.__name__ + ": ",
        lib.__version__,
        "\n" if lib.__name__ == "pytorch_lightning" else "",
    )

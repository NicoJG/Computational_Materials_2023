# %%
import os
if os.path.basename(os.getcwd()) == "task2":
    os.chdir("..")
    print("Changed working directory!")

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read

trajectory = read("../task1/someDynamics.traj")


# %%

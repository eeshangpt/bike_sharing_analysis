# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from os import getcwd
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
PRJ_DIR = getcwd()

# %%
DATA_DIR = join(PRJ_DIR, "data")

# %%
data = pd.read_csv(join(DATA_DIR, "day.csv"))

# %%
data.head()

# %%
with open(join(DATA_DIR, "Readme.txt"), 'r') as f:
    data_dictonary = f.read()

# %%
print(data_dictonary)

# %%

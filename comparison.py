import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
import extra_funcs
import shap
from collections import defaultdict
from pathlib import Path
import joblib


y_label = "outcome_A"

df1 = extra_funcs.load_entire_dataframe("NBI_Predict_TXA_recepter.csv")
X1 = extra_funcs.df_to_X(df1)
y1 = df1.loc[:, y_label]


df2 = extra_funcs.load_entire_dataframe(
    "Dtasætround2_NBI_Predict_PrimæreTXA_14_17_MASTER_WORK_1_Hb_Recepter_knee.csv"
)
X2 = extra_funcs.df_to_X(df2)
y2 = df2.loc[:, y_label]

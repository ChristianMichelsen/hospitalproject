#%%

import numpy as np
import pandas as pd


import extra_funcs

cols = np.array(
    [
        "sex",
        "joint",
        "hb",
        "kidney",
        "height",
        "weight",
        "bmi",
        "smoking",
        "alcohol",
        "civil_status",
        "walking_tool",
        "dm_type",
        "cardiac_disease",
        "pulmonary_disease",
        "psd_knee",
        "cerebral_attack",
        "group_ak",
        "steroid",
        "group_card",
        "group_psych",
        "group_resp",
        "cholesterol_medicine",
        "antirheumatika",
        "hypertens",
    ]
)



df_full = extra_funcs.load_entire_dataframe()

df_full.loc[:, cols]


# ["DM_type", "Pulmonary_disease", "Gangredskab", osv.]

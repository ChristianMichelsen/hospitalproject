#%%

from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

import extra_funcs

#%%

y_label = "outcome_A"

PPF = 0.20

cfg = dict()

cfg["force_HPO"] = False
cfg["n_trials"] = 1000
cfg["n_jobs"] = 6
cfg["optimize"] = "AUC"
cfg["FL_str"] = "no_FL"
cfg["PPF"] = PPF

use_FL = False

cfg_str = extra_funcs.cfg_to_str(cfg)
cfg_str_with_PPF = f"{cfg_str}__{PPF:.2f}"


#%%

dict_variables = [
    "data",
    "y_pred_proba",
    "y_pred_proba_train",
    "y_pred",
    "results",
    "ROC",
    "PR",
    "PRG",
    "models",
]

dicts = {}
for k in dict_variables:
    dicts[k] = defaultdict()


#%%



#%%


extra_funcs.add_ML_model(
    cfg=cfg,
    dicts=dicts,
    y_label=y_label,
    key=f"ML",
    use_FL=use_FL,
    name=f"{y_label}__ML__{cfg_str}",
    PPF_cut=PPF,
)


extra_funcs.add_ML_model(
    cfg=cfg,
    dicts=dicts,
    y_label=y_label,
    key=f"ML_26",
    use_FL=use_FL,
    name=f"{y_label}__ML__{cfg_str}",
    PPF_cut=PPF,
    df=df,
    X=X,
    y=y,
)


data_ROC = extra_funcs.extract_data_ROC(dicts)
data_df = extra_funcs.extract_data_df(dicts)
data_risc_scores = extra_funcs.extract_data_risc_scores(dicts)

df_results = pd.DataFrame.from_dict(dicts["results"], orient="index")

# fmt:off
metrics = [
        "TP", "FP", "FN", "TN",
        # "TPR_mean", "PPV_mean", "FPR_mean", "ACC_mean", "PPF_mean",
        "TPR_mean", "PPV_mean", "ACC_mean",
        "F1", "MCC",
        # "RG", "PG",
        "ROC_AUC", "PR_AUC", "PRG_AUC",
        "cutoff",
    ]
# fmt:on

df_results_save = df_results.loc[:, metrics]


print("plotting stuff")
# reload(extra_funcs)

extra_funcs.make_ROC_curves(
    data_risc_scores,
    data_ROC,
    cfg_str_with_PPF,
    include_ML__exclude_age=True,
    cuts=[(PPF - 0.05, PPF + 0.05)],
)

#%%
from collections import defaultdict

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import extra_funcs

#%%

key = f"ML__26"
y_label = "outcome_B"
# y_label = "outcome_A"
PPF = 0.20

cfg = dict()

cfg["force_HPO"] = False
cfg["n_trials"] = 1000
cfg["n_jobs"] = 6


cfg["optimize"] = "AUC"  # "TPR", "focal_loss", "AUC", "average_precision"
cfg["FL_str"] = "no_FL"
cfg["PPF"] = PPF
cfg["add_ML_26"] = True

cfg_str = extra_funcs.cfg_to_str(cfg)
cfg_str_with_PPF = f"{cfg_str}__{PPF:.2f}"

# name = f"{y_label}__ML__{cfg_str}"
name = f"{y_label}__ML__26__{cfg_str}"

# x=x


#%%


def get_lgb_params(name):
    filename_optuna = f"./models/optuna__{name}.pkl"
    study = joblib.load(filename_optuna)
    trial = study.best_trial
    d_optuna_all = dict(trial.params)
    d_optuna_all["num_boost_round"] = trial.user_attrs["num_boost_round"]
    params = extra_funcs.params_from_optuna(d_optuna_all)
    return params


def get_lgb_num_boost_round(name):
    filename_optuna = f"./models/optuna__{name}.pkl"
    study = joblib.load(filename_optuna)
    trial = study.best_trial
    d_optuna_all = dict(trial.params)
    num_boost_round = trial.user_attrs["num_boost_round"]
    return num_boost_round


def get_dataset_method(name):
    filename_optuna = f"./models/optuna__{name}.pkl"
    study = joblib.load(filename_optuna)
    trial = study.best_trial
    d_optuna_all = dict(trial.params)
    dataset_method = d_optuna_all["dataset"]
    return dataset_method


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


params = get_lgb_params(name)
num_boost_round = get_lgb_num_boost_round(name)
dataset_method = get_dataset_method(name)


#%%


cols_26_ordered = extra_funcs.load_cols_26(y_label)

df = extra_funcs.load_entire_dataframe()
X_full, y = extra_funcs.df_to_X_y(df, y_label)
X = X_full.loc[:, cols_26_ordered]


(
    columns_cat_all,
    columns_cat_subset,
) = extra_funcs.get_numerical_and_categorical_features(X)

# test on 2017
mask_test = "2017-01-01" <= df["date"]
df_test = df.loc[mask_test].copy()
X_test = X.loc[mask_test].copy()
y_test = y.loc[mask_test].copy()

# 6 validation time intervals of 2 months in 2016
end_times = (
    [f"2014-01-{i:02d}" for i in range(3, 30, 1)]
    + [f"2014-{i:02d}-01" for i in range(2, 13, 1)]
    + [f"2015-{i:02d}-01" for i in range(1, 13, 1)]
    + [f"2016-{i:02d}-01" for i in range(1, 13, 1)]
    + ["2017-01-01"]
)

Ns = []
TPRs = []

for end_time in tqdm(end_times):

    # train and validate on < 2017
    mask_train_val = df["date"] < end_time
    df_train_val = df.loc[mask_train_val].copy()
    X_train_val = X.loc[mask_train_val].copy()
    y_train_val = y.loc[mask_train_val].copy()
    N = len(X_train_val)
    Ns.append(N)

    dataset_train = extra_funcs.pandas_to_lgb_dataset(
        X_train_val,
        y_train_val,
        columns_cat_all=columns_cat_all,
        columns_cat_subset=columns_cat_subset,
        method=dataset_method,
        init_score=None,
    )

    model = lgb.train(
        params,
        dataset_train,
        num_boost_round=num_boost_round,
        # verbose_eval=100,
    )

    y_pred_proba = model.predict(X_test)
    y_pred_proba_train = model.predict(X_train_val)

    y_true = y_test

    cutoff = extra_funcs.compute_cutoff(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        PPF_cut=PPF,
        N=1_000_000,
    )

    y_pred = y_pred_proba >= cutoff

    d_cm = extra_funcs.compute_confusion_matrix(y_true.values, y_pred)
    d_out = {}
    for measure in ["TP", "TN", "FN", "FP"]:
        d_out[measure] = d_cm[measure]
    TPR = d_cm["TP"] / (d_cm["TP"] + d_cm["FN"])

    TPRs.append(TPR)


# %%

Ns = np.array(Ns)
TPRs = np.array(TPRs)


#%%

fig, axes = plt.subplots(figsize=(15, 6), ncols=2)
axes[0].plot(Ns, TPRs)
axes[0].axhline(TPRs[-1], ls="--", c="k", alpha=0.2)
axes[0].set(xlabel="N", ylabel="TPR", xlim=(0, None))
axes[1].plot(Ns, TPRs)
axes[1].axhline(TPRs[-1], ls="--", c="k", alpha=0.2)
axes[1].set(xlabel="N", ylabel="TPR", xscale="log", xlim=(10, None))
fig.suptitle("True Positive Rate as function of number of patients", fontsize=16)
fig.savefig(f"./figures/TPR_number_of_patients__{y_label}.pdf")

# %%

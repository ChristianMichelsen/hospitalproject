import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from copy import copy
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.metrics import (
    auc,
    f1_score,
    fbeta_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    recall_score,
    classification_report,
    precision_recall_curve,
    matthews_corrcoef,
    cohen_kappa_score,
)

import platform
import os


def is_hep():
    return platform.system() == "Linux" and os.cpu_count() > 17


#%%

# filename = "Dtasæt_NBI_Predict_PrimæreTXA_16_17_MASTER_ WORK.csv"
# filename = "Dtasætround2_NBI_Predict_PrimæreTXA_14_17_MASTER_WORK_1_Hb.csv"
# filename = "Dtasætround2_NBI_Predict_PrimæreTXA_14_17_MASTER_WORK_1_Hb_LOS.csv"
# filename_csv = "Dtasætround3_NBI_Predict_PrimæreTXA_14_17_M ASTER_WORK_1_Hb.csv"
# filename_csv = "NBI_Predict_TXA_recepter.csv"
filename_csv = (
    "Dtasætround2_NBI_Predict_PrimæreTXA_14_17_MASTER_WORK_1_Hb_Recepter_knee.csv"
)

d_columns_rename = {
    "D_ODTO": "date",
    "Køn": "sex",
    "Civilstatus": "civil_status",
    "Height": "height",
    "Weight": "weight",
    "Hb": "hb",
    "Rygning": "smoking",
    "Alkohol": "alcohol",
    "Gangredskab": "walking_tool",
    "Udhvilet": "rested",
    "Snorken": "snore",
    "DM_type": "dm_type",  # diabetes type
    # "Hypertension_ja_ell_recept": "hypertension_yes_or_prescription",
    "Hyperkolesterol": "hyper_colesterol",
    "Cardiac_disease": "cardiac_disease",
    "Pulmonary_disease": "pulmonary_disease",
    "Cerebral_attack": "cerebral_attack",
    "Tidl_VTE": "previous_vte",  # blodprop / årebetædndelse
    "Fam_VTE": "family_vte",
    "Cancer": "cancer",
    "Nyre": "kidney",  # nyresvigt
    # "PsD": "psd",  # psykopharmaka
    "Led": "joint",
    "Alder": "age",
    "BMI": "bmi",
    # "PotentAK": "potent_ak",  # blodfortyndende behandling, renset op, stærkeste medicin
    "Årstal": "year",
    "Hospital": "hospital",
    "Medical_outcome": "medical_outcome_old",  # more or less than 4 days in hospital (or re-indlæggelse)
    "Medical_outcome_inkl_obs_diag": "outcome_A",  # more or less than 4 days in hospital (or re-indlæggelse)
    "Medicaloutcome3": "outcome_B",
    "F_post_op_liggetid": "length_of_stay",
    "Risikoscore1": "risc_score1",
    "Antirheumatika": "antirheumatika",
    "Gruppering_AK": "group_ak",
    "Grupperinger_Card": "group_card",
    "Grupperinger_Psyk": "group_psych",
    "Grupperinger_Resp": "group_resp",
    "Steroid": "steroid",
    "Total_antal_Recepter": "N_total_prescriptions",
    "kolesterolmedicin": "cholesterol_medicine",
    "PsD_knee": "psd_knee",
    "hypertens": "hypertens",
}


d_translate = {
    "sex": r"$\mathrm{Gender}$",
    "civil_status": r"$\mathrm{Living \,\, alone}$",
    "height": r"$\mathrm{Height}$",
    "weight": r"$\mathrm{Weight}$",
    "hb": r"$\mathrm{Hb}$",
    "smoking": r"$\mathrm{Smoking}$",
    "alcohol": r"$\mathrm{Alcohol}$",
    "walking_tool": r"$\mathrm{Walking \,\, aid}$",
    "rested": r"$\mathrm{Rested}$",
    "snore": r"$\mathrm{Snore}$",
    "dm_type": r"$\mathrm{Diabetes}$",
    # "hypertension_yes_or_prescription": r"$\mathrm{Hypertension}$",
    "hyper_colesterol": r"$\mathrm{Hyper \,\, cholesterol}$",
    "cardiac_disease": r"$\mathrm{Cardiac}$",
    "pulmonary_disease": r"$\mathrm{Pulmonary}$",
    "cerebral_attack": r"$\mathrm{Cerebral}$",
    "previous_vte": r"$\mathrm{Prev. \,\, VTE}$",
    "family_vte": r"$\mathrm{Fam. \,\, VTE}$",
    "cancer": r"$\mathrm{Cancer}$",
    "kidney": r"$\mathrm{Kidney}$",
    # "psd": r"$\mathrm{PSD}$",
    "psd_knee": r"$\mathrm{PSD \,\, knee}$",
    "joint": r"$\mathrm{Joint}$",
    "age": r"$\mathrm{Age}$",
    "bmi": r"$\mathrm{BMI}$",
    # "potent_ak": r"$\mathrm{AK \,\, (potent)}$",
    "year": r"$\mathrm{Year}$",
    # "month": r"$\mathrm{Month}$",
    # "day_of_week": r"$\mathrm{Week \,\, day}$",
    # "day_of_month": r"$\mathrm{Day \,\, of \,\, month}$",
    "antirheumatika": r"$\mathrm{Antirheumatika}$",
    "group_ak": r"$\mathrm{Prescribed \,\, anticoagulants}$",
    "group_card": r"$\mathrm{Prescribed \,\, cardiac \,\, drugs}$",
    "group_psych": r"$\mathrm{Psychotropics}$",
    "group_resp": r"$\mathrm{Respiratory \,\, drugs}$",
    "steroid": r"$\mathrm{Steroid}$",
    "N_total_prescriptions": r"$N_\mathrm{prescriptions}$",
    "cholesterol_medicine": r"$\mathrm{Prescribed \,\, anticholesterols}$",
    "hypertens": r"$\mathrm{hypertens}$",
}


def add_date_info_to_df(df):
    # df["month"] = df["date"].dt.month
    # df["day_of_week"] = df["date"].dt.day_of_week
    # df["day_of_month"] = df["date"].dt.day
    df.loc[:, "year"] = df["date"].dt.day_of_year / 366 + df["date"].dt.year
    return df


non_integer_columns = ["hb", "bmi", "year", "date"]


def make_variables_ints(df):
    variables = [col for col in df.columns if not col in non_integer_columns]
    df[variables] = df[variables].astype(int)
    return df


def remove_date_nans(df):
    return df[~df["date"].isna()]


def add_cuts(df):
    return df.query("30 <= weight <= 250 & 100 <= height <= 210")


def parse_dates(df):
    df.loc[:, "D_ODTO"] = pd.to_datetime(df["D_ODTO"], format="%m/%d/%Y 0:00:00")
    return df


def remove_to_much_nans(df):
    mask = df.isna().sum(axis=1) <= 5
    return df[mask]


def load_entire_dataframe(filename=filename_csv):

    df = (
        pd.read_csv(filename, sep=";", decimal=",", na_values=" ")
        .pipe(parse_dates)
        .rename(columns=d_columns_rename)
        .sort_values(by=["date", "height", "bmi", "hb"])
        .pipe(add_cuts)
        .pipe(remove_date_nans)
        .pipe(add_date_info_to_df)
        # .pipe(remove_to_much_nans)
        .reset_index(drop=True)
    )

    return df


def df_to_X(df):

    drop_columns = [
        "date",
        "Anæmi",
        "AK_beh",
        # "recept_PsD",
        "outcome_A",
        "outcome_B",
        "medical_outcome_old",
        "risc_score1",
        "cutoff6",
        "cutoff5",
        "length_of_stay",
        "hospital",
    ]

    if "Komb" in df.columns:
        drop_columns.append("Komb")

    if "ID" in df.columns:
        drop_columns.append("ID")

    X = df.drop(columns=drop_columns)
    return X


def df_to_X_y(df, y_label):
    X = df_to_X(df)
    y = df.loc[:, y_label]
    return X, y


numerical_columns = [
    "bmi",
    "year",
    "weight",
    "age",
    "hb",
    "height",
    "N_total_prescriptions",
]


def get_numerical_and_categorical_features(X):

    categorical_features = [col for col in X.columns if col not in numerical_columns]

    columns_cat_subset = X.nunique()[categorical_features]
    columns_cat_subset = columns_cat_subset[columns_cat_subset >= 3]
    columns_cat_subset = list(columns_cat_subset.index)

    return categorical_features, columns_cat_subset


#%%


def cfg_to_str(cfg):
    return f"{cfg['optimize']}__{cfg['n_trials']}__{cfg['FL_str']}"
    # return f"{cfg['optimize']}__{cfg['n_trials']}__{cfg['FL_str']}__{cfg['PPF']}"


#%%


def get_train_test_splits(df_train_val, time_intervals_val):

    df = df_train_val

    # df["idx"] = range(len(df))

    cv_splits = []
    for i, interval in enumerate(time_intervals_val[:-1]):
        # break
        idx_train = df.index[df["date"] < interval]
        mask_test = (interval <= df["date"]) & (df["date"] < time_intervals_val[i + 1])
        idx_test = df.index[mask_test]
        cv_splits.append((idx_train.values, idx_test.values))
    return cv_splits


def get_data(y_label, exclude=None, include=None, filename=filename_csv):

    if exclude is not None and include is not None:
        raise AssertionError(f"exclude and include cannot both be set.")

    # load data
    df = load_entire_dataframe(filename=filename)
    X, y = df_to_X_y(df, y_label)

    if exclude is not None:

        if isinstance(exclude, str):
            exclude = [exclude]

        for column in exclude:
            if column in X.columns:
                X = X.drop(columns=column)

    elif include is not None:
        X = X.loc[:, include]

    # find categorical features / columns / variables
    columns_cat_all, columns_cat_subset = get_numerical_and_categorical_features(X)

    # test on 2017
    mask_test = "2017-01-01" <= df["date"]
    df_test = df.loc[mask_test].copy()
    X_test = X.loc[mask_test].copy()
    y_test = y.loc[mask_test].copy()

    # train and validate on < 2017
    mask_train_val = df["date"] < "2017-01-01"
    df_train_val = df.loc[mask_train_val].copy()
    X_train_val = X.loc[mask_train_val].copy()
    y_train_val = y.loc[mask_train_val].copy()

    # 6 validation time intervals of 2 months in 2016
    time_intervals_val = [f"2016-{i:02d}-01" for i in range(1, 12, 2)] + ["2017-01-01"]

    cv_splits_validation = get_train_test_splits(df_train_val, time_intervals_val)

    X_train_val_imputed, imputer = impute(X_train_val)
    X_test_imputed, _ = impute(X_test, imputer=imputer)

    scaler = StandardScaler()
    X_train_val_imputed_scaled = scaler.fit_transform(X_train_val_imputed)
    X_train_val_imputed_scaled = pd.DataFrame(
        X_train_val_imputed_scaled,
        columns=X_train_val_imputed.columns,
        index=X_train_val_imputed.index,
    )
    X_test_imputed_scaled = scaler.transform(X_test_imputed)
    X_test_imputed_scaled = pd.DataFrame(
        X_test_imputed_scaled,
        columns=X_test_imputed.columns,
        index=X_test_imputed.index,
    )

    d_data = {}
    d_data["X"] = X
    d_data["X_train_val"] = X_train_val
    d_data["X_train_val_imputed"] = X_train_val_imputed
    d_data["X_test"] = X_test
    d_data["X_test_imputed"] = X_test_imputed

    d_data["y"] = y
    d_data["y_train_val"] = y_train_val
    d_data["y_test"] = y_test

    d_data["df"] = df
    d_data["df_train_val"] = df_train_val
    d_data["df_test"] = df_test

    d_data["columns_cat_all"] = columns_cat_all
    d_data["columns_cat_subset"] = columns_cat_subset

    d_data["cv_splits_validation"] = cv_splits_validation
    d_data["imputer"] = imputer

    d_data["X_train_val_imputed_scaled"] = X_train_val_imputed_scaled
    d_data["X_test_imputed_scaled"] = X_test_imputed_scaled
    d_data["scaler"] = scaler

    return d_data


#%%


def get_table(X):

    numerical_columns_no_year = numerical_columns.copy()
    numerical_columns_no_year.remove("year")
    if not "N_total_prescriptions" in X.columns:
        numerical_columns_no_year.remove("N_total_prescriptions")

    categories = [col for col in X.columns if col not in numerical_columns_no_year]

    d_table = {}
    for numeric in numerical_columns_no_year:
        median = X[numeric].median()
        IQR = X[numeric].quantile(0.25), X[numeric].quantile(0.75)
        d_table[numeric] = f"{median:.1f} ({IQR[0]:.1f}-{IQR[1]:.1f})"

    for category in categories:
        counts = X[category].value_counts(normalize=False).sort_index()
        freqs = X[category].value_counts(normalize=True).sort_index()
        for cat in counts.index:
            count = counts.loc[cat]
            freq = freqs.loc[cat]
            d_table[f"{category} == {cat}"] = f"{count} ({freq:.1%})"

    return d_table


def get_table_df(y_label):

    d_data = get_data(y_label)
    X = d_data["X"]
    y = d_data["y"]

    d_table = {}
    d_table["Signal"] = get_table(X.loc[y == 1])
    d_table["Background"] = get_table(X.loc[y == 0])

    df_table = pd.DataFrame(d_table)
    df_table["Missing"] = ["-"] * len(df_table)

    for col, val in (X.isnull().sum() / len(X)).items():
        df_table.loc[col, "Missing"] = f"{val:.1%}"

    return df_table.sort_index().fillna("-")


def extract_data_df(dicts):
    for key in dicts["data"].keys():
        data_df = dicts["data"][key]["df"]
        break
    return data_df


def extract_all_data_df(dicts):
    for key in dicts["data"].keys():
        data_all = dicts["data"][key]
        break
    return data_all


#%%


def get_lgb_datasets(d_data):

    d_lgb_datasets = {}

    methods = ["simple", "categorical_all", "categorical_subset"]

    for X, name in zip(
        [d_data["X_train_val"], d_data["X_train_val_imputed"]],
        ["", "imputed_"],
    ):

        for method in methods:

            d_lgb_datasets[f"{name}{method}"] = pandas_to_lgb_dataset(
                X,
                d_data["y_train_val"],
                columns_cat_all=d_data["columns_cat_all"],
                columns_cat_subset=d_data["columns_cat_subset"],
                method=method,
            )

    return d_lgb_datasets


def pandas_to_lgb_dataset(
    X,
    y,
    columns_cat_all,
    columns_cat_subset,
    method,
    init_score=None,
):

    if "simple" in method:

        dataset = lgb.Dataset(
            X.values,
            label=y.values,
            feature_name=list(X.columns),
            free_raw_data=False,
            init_score=init_score,
        )

    elif "categorical_all" in method:
        dataset = lgb.Dataset(
            X,
            label=y,
            categorical_feature=columns_cat_all,
            free_raw_data=False,  # needed for categorical
            init_score=init_score,
        )

    elif "categorical_subset" in method:
        dataset = lgb.Dataset(
            X,
            label=y,
            categorical_feature=columns_cat_subset,
            free_raw_data=False,  # needed for categorical
            init_score=init_score,
        )

    return dataset


#%%


def impute(X, imputer=None, round_to_int=True):
    """
    First fit training set, then transform training and validation.
    Afterwards fit training and validation set, then transform test set
    """

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imputed_raw = imputer.fit_transform(X)
    else:
        X_imputed_raw = imputer.transform(X)

    X_imputed = pd.DataFrame(
        X_imputed_raw,
        index=X.index,
        columns=X.columns,
    )

    for col in X_imputed.columns:
        if round_to_int and X[col].nunique() < 10:
            X_imputed[col] = X_imputed[col].astype(int)
        else:
            X_imputed[col] = X_imputed[col].astype(X[col].dtypes.name)

    return X_imputed, imputer


#%%


def create_length_of_stay_sig_bkg(data_df):

    titles = [
        r"$\mathrm{Outcome \,\, A}$",
        r"$\mathrm{Outcome \,\, B}$",
    ]

    for i_ax, (y_label, df) in enumerate(data_df.items()):

        fig, ax = plt.subplots(figsize=(5, 5))

        df_sig = df.query(f"{y_label} == 1")
        df_bkg = df.query(f"{y_label} == 0")

        N_days = 20
        kwargs = dict(
            bins=N_days,
            range=(0, N_days),
            histtype="step",
            density=True,
            lw=2,
            # align="left",
        )

        ax.hist(
            df_sig.length_of_stay,
            label="Positive",
            **kwargs,
        )
        ax.hist(
            df_bkg.length_of_stay,
            label="Negative",
            **kwargs,
        )

        ax.set(
            xlabel="Length of stay [days]",
            ylabel="Counts (normalized)",
            # xlim=(0 - 0.5, N_days - 0.5),
            xlim=(0, N_days),
        )
        ax.legend()
        ax.xaxis.get_major_locator().set_params(integer=True)

        ax.set_title(titles[i_ax], pad=10)

        fig.tight_layout()

        filename = f"./figures/LOS__{y_label}"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename + ".pdf", dpi=300)
        fig.savefig(filename.replace("figures/", "figures/pngs/") + ".png", dpi=600)


def extract_data_risc_scores(dicts):
    d_risc_scores = {}
    for key in dicts["data"].keys():
        d_risc_scores[key] = {
            "y_test": dicts["data"][key]["y_test"],
            "y_pred_proba": dicts["y_pred_proba"][key],
            "cutoff": dicts["results"][key]["cutoff"],
        }
    return d_risc_scores


# #%%

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import special


class FocalLoss:
    # https://maxhalford.github.io/blog/lightgbm-focal-loss/

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = minimize_scalar(
            lambda p: self(y_true, p).sum(), bounds=(0, 1), method="bounded"
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return "focal_loss", self(y, p).mean(), is_higher_better


#%%


from numba import njit
import numba as nb
from numba.core import types
from numba.typed import Dict


@njit
def compute_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for i in range(len(y_true)):
        cm[int(y_true[i])][int(y_pred[i])] += 1
    d_cm = {}
    d_cm["TP"] = cm[1, 1]
    d_cm["FN"] = cm[1, 0]
    d_cm["FP"] = cm[0, 1]
    d_cm["TN"] = cm[0, 0]
    d_cm["sum"] = cm.sum()
    return d_cm


@njit
def compute_PPF(d_cm):
    return (d_cm["TP"] + d_cm["FP"]) / d_cm["sum"]


@njit
def compute_TPR(d_cm):
    return d_cm["TP"] / (d_cm["TP"] + d_cm["FN"])


@njit
def _compute_cutoffs(y_pred_proba):
    x = np.sort(y_pred_proba)
    midpoints = (x[1:] + x[:-1]) / 2
    cutoffs = np.zeros(len(midpoints) + 2)
    for i, midpoint in enumerate(midpoints):
        cutoffs[i + 1] = midpoint
    cutoffs[-1] = 1
    # cutoffs = np.append(cutoffs, 1)
    return cutoffs


@njit
def nb_compute_cutoff_PPF_TPR(
    y_true,
    y_pred_proba,
    PPF_cut,
    minimum=-1,
    maximum=-1,
    N=1_000_000,
):

    if minimum == -1:
        minimum = np.nanmin(y_pred_proba)

    if maximum == -1:
        maximum = np.nanmax(y_pred_proba)

    if N < len(y_pred_proba):
        cutoffs = np.linspace(minimum, maximum, N)
    else:
        # x = np.sort(y_pred_proba)
        # midpoints = (x[1:] + x[:-1]) / 2
        # cutoffs = np.insert(midpoints, 0, 0)
        # cutoffs = np.append(cutoffs, 1)
        cutoffs = _compute_cutoffs(y_pred_proba)

    for cutoff in cutoffs:
        y_pred = y_pred_proba >= cutoff
        d_cm = compute_confusion_matrix(y_true, y_pred)
        PPF = compute_PPF(d_cm)
        if PPF < PPF_cut:
            TPR = compute_TPR(d_cm)
            # break
            return cutoff, PPF, TPR

    return cutoff, PPF, np.nan


def compute_cutoff_PPF_TPR(
    y_true,
    y_pred_proba,
    PPF_cut,
    minimum=-1,
    maximum=-1,
    N=1_000_000,
):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values
    return nb_compute_cutoff_PPF_TPR(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        PPF_cut=PPF_cut,
        minimum=minimum,
        maximum=maximum,
        N=N,
    )


def compute_cutoff(
    y_true,
    y_pred_proba,
    PPF_cut,
    minimum=-1,
    maximum=-1,
    N=1_000_000,
):
    return compute_cutoff_PPF_TPR(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        PPF_cut=PPF_cut,
        minimum=minimum,
        maximum=maximum,
        N=N,
    )[0]


def lgb_eval_TPR_given_PPF(y_pred_proba, train_data):
    y_true = train_data.get_label()
    is_higher_better = True
    _, _, TPR = nb_compute_cutoff_PPF_TPR(y_true, y_pred_proba)
    return "TPR", TPR, is_higher_better


def binomial_mean_std(x, N):
    mean = x / N
    std = np.sqrt(mean * (1 - mean) / N)
    return mean, std


def _compute_single_metric(d_cm, measure):

    if measure.upper() == "TPR":
        x = d_cm["TP"]
        N = d_cm["TP"] + d_cm["FN"]

    elif measure.upper() == "PPV":
        x = d_cm["TP"]
        N = d_cm["TP"] + d_cm["FP"]

    elif measure.upper() == "FPR":
        x = d_cm["FP"]
        N = d_cm["FP"] + d_cm["TN"]

    elif measure.upper() == "ACC":
        x = d_cm["TP"] + d_cm["TN"]
        N = d_cm["sum"]

    elif measure.upper() == "PPF":
        x = d_cm["TP"] + d_cm["FP"]
        N = d_cm["sum"]

    return binomial_mean_std(x, N)


from prg import prg


def _compute_performance_measures(y_true, y_pred):

    d_cm = compute_confusion_matrix(y_true, y_pred)

    d_out = {}

    for measure in ["TP", "TN", "FN", "FP"]:
        d_out[measure] = d_cm[measure]

    for measure in ["TPR", "PPV", "FPR", "ACC", "PPF"]:
        mean, std = _compute_single_metric(d_cm, measure)
        d_out[f"{measure}_mean"], d_out[f"{measure}_std"] = mean, std

    d_out["F1"] = f1_score(y_true, y_pred)
    d_out["MCC"] = matthews_corrcoef(y_true, y_pred)

    RG = prg.recall_gain(d_out["TP"], d_out["FN"], d_out["FP"], d_out["TN"])
    PG = prg.precision_gain(d_out["TP"], d_out["FN"], d_out["FP"], d_out["TN"])
    d_out["RG"] = RG
    d_out["PG"] = PG

    return d_out


from sklearn.metrics import roc_auc_score, average_precision_score


def compute_PRG_AUC(y_true, y_pred_proba):
    labels = y_true
    prg_curve = prg.create_prg_curve(y_true, y_pred_proba)
    auprg = prg.calc_auprg(prg_curve)
    return auprg


def compute_performance_measures(dicts, key, PPF_cut, cutoff=None):

    y_true = dicts["data"][key]["y_test"]
    y_pred_proba = dicts["y_pred_proba"][key]

    if cutoff is None:
        cutoff = compute_cutoff(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            PPF_cut=PPF_cut,
            N=1_000_000,
        )

    y_pred = y_pred_proba >= cutoff

    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values

    d_out = dict(_compute_performance_measures(y_true, y_pred))

    for measure in ["TP", "FP", "FN", "TN"]:
        d_out[measure] = int(d_out[measure])

    mask = ~np.isnan(y_pred_proba)
    d_out["ROC_AUC"] = roc_auc_score(y_true[mask], y_pred_proba[mask])
    d_out["PR_AUC"] = average_precision_score(y_true[mask], y_pred_proba[mask])
    d_out["PRG_AUC"] = compute_PRG_AUC(y_true[mask], y_pred_proba[mask])
    d_out["cutoff"] = cutoff

    dicts["results"][key] = d_out

    fpr, tpr, threshold_roc = roc_curve(y_true[mask], y_pred_proba[mask])
    dicts["ROC"][key] = {"fpr": fpr, "tpr": tpr, "threshold_roc": threshold_roc}

    precision, recall, threshold_pr = precision_recall_curve(
        y_true[mask],
        y_pred_proba[mask],
    )
    dicts["PR"][key] = {
        "precision": precision,
        "recall": recall,
        "threshold_pr": threshold_pr,
    }

    prg_curve = prg.create_prg_curve(y_true[mask], y_pred_proba[mask])
    dicts["PRG"][key] = {
        "recall_gain": prg_curve["recall_gain"],
        "precision_gain": prg_curve["precision_gain"],
    }


#%%


# #%%

from sklearn.preprocessing import MinMaxScaler


def add_model_age_only(dicts, y_label, key, PPF_cut):

    d_data = get_data(y_label)
    X_test = d_data["X_test"]
    dicts["data"][key] = d_data

    scaler = MinMaxScaler()

    X_test_np = X_test["age"].values.reshape(-1, 1)
    dicts["y_pred_proba"][key] = scaler.fit_transform(X_test_np)[:, 0]

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)


#%%


def add_model_risc_score1(dicts, y_label, key, PPF_cut):
    d_data = get_data(y_label)
    dicts["data"][key] = d_data

    df_test = d_data["df_test"]
    dicts["y_pred_proba"][key] = df_test["risc_score1"] / df_test["risc_score1"].max()

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)


#%%

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def add_model_LR(dicts, y_label, key, PPF_cut, exclude=None, include=None):

    d_data = get_data(y_label, exclude=exclude, include=include)
    dicts["data"][key] = d_data

    clf = LogisticRegression(random_state=0, max_iter=10_000).fit(
        d_data["X_train_val_imputed_scaled"], d_data["y_train_val"]
    )

    dicts["y_pred_proba"][key] = clf.predict_proba(d_data["X_test_imputed_scaled"])[
        :, 1
    ]
    dicts["y_pred_proba_train"][key] = clf.predict_proba(
        d_data["X_train_val_imputed_scaled"]
    )[:, 1]

    dicts["models"][key] = clf

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)


#%%

import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
import joblib


#%%

#%%


def params_from_optuna(d_optuna_all):

    d_params = {}

    d_params_base = {
        "objective": "binary",
        "boosting": "gbdt",
        "verbose": -1,
        "is_unbalance": False,
        "bagging_freq": 1,
    }

    d_params["optuna_all"] = {
        **d_params_base,
        **{
            # "boosting": d_optuna_all["boosting_type"],
            "max_depth": d_optuna_all["max_depth"],
            "num_leaves": d_optuna_all["num_leaves"],
            "scale_pos_weight": d_optuna_all["scale_pos_weight"],
            "min_child_weight": d_optuna_all["min_child_weight"],
            "colsample_bytree": d_optuna_all["colsample_bytree"],
            "bagging_fraction": d_optuna_all["bagging_fraction"],
        },
    }

    params = d_params["optuna_all"]
    return params


def fl_from_optuna(d_optuna_all, d_data):

    y_train_val = d_data["y_train_val"]

    fl = FocalLoss(gamma=d_optuna_all["fl_gamma"], alpha=d_optuna_all["fl_alpha"])

    init_score = np.full_like(y_train_val, fl.init_score(y_train_val), dtype=float)
    fobj = fl.lgb_obj

    return fl, init_score, fobj


def dataset_train_from_optuna(d_optuna_all, d_data, init_score=None):

    if "imputed" in d_optuna_all["dataset"]:
        d_data["X_train_val"] = d_data["X_train_val_imputed"]
        d_data["X_test"] = d_data["X_test_imputed"]

    dataset_train = pandas_to_lgb_dataset(
        d_data["X_train_val"],
        d_data["y_train_val"],
        columns_cat_all=d_data["columns_cat_all"],
        columns_cat_subset=d_data["columns_cat_subset"],
        method=d_optuna_all["dataset"],
        init_score=init_score,
    )

    return dataset_train


#%%

import types


class FLModel:
    def __init__(self, d_optuna_all, d_data):
        self.d_optuna_all = d_optuna_all
        self.d_data = d_data
        self._setup()

    def _setup(self):
        self.params = params_from_optuna(self.d_optuna_all)
        self.fl, self.init_score, self.fobj = self._fl_from_optuna()
        self.dataset_train = self._dataset_train_from_optuna()

    def _fl_from_optuna(self):
        fl, init_score, fobj = fl_from_optuna(self.d_optuna_all, self.d_data)
        return fl, init_score, fobj

    def _dataset_train_from_optuna(self):
        dataset_train = dataset_train_from_optuna(
            self.d_optuna_all, self.d_data, self.init_score
        )
        return dataset_train

    def fit(self):
        self.model = lgb.train(
            self.params,
            self.dataset_train,
            num_boost_round=self.d_optuna_all["num_boost_round"],
            verbose_eval=100,
            fobj=self.fobj,
        )
        return self

    def predict(self, X):
        return special.expit(
            self.fl.init_score(self.d_data["y_train_val"]) + self.model.predict(X)
        )

    def __getstate__(self):
        return {
            "d_optuna_all": self.d_optuna_all,
            "d_data": self.d_data,
            "model": self.model,
        }

    def __setstate__(self, d):
        self.d_optuna_all = d["d_optuna_all"]
        self.d_data = d["d_data"]
        self.model = d["model"]
        self._setup()

    # def get_model(self):

    #     model = self.model
    #     alpha = self.d_optuna_all["fl_alpha"]
    #     gamma = self.d_optuna_all["fl_gamma"]
    #     y_train_val = self.d_data["y_train_val"]

    #     def predict_proba(self, X):
    #         fl = FocalLoss(gamma=gamma, alpha=alpha)
    #         pred = special.expit(fl.init_score(y_train_val) + model.predict(X))
    #         return np.array([1 - pred, pred]).T

    #     model.predict_proba = types.MethodType(predict_proba, model)
    #     return model

    # def predict_proba(self, X):
    #     pred = self.predict(X)
    #     return np.array([1 - pred, pred]).T


#%%


def add_ML_model(
    cfg,
    dicts,
    y_label,
    key,
    name,
    PPF_cut,
    use_FL=False,
    exclude=None,
    include=None,
):

    print(f"\n\nFitting ML model {key}. \n\n")

    cfg = copy(cfg)
    cfg["exclude"] = exclude

    d_data = get_data(y_label, exclude=exclude, include=include)
    dicts["data"][key] = d_data

    # define lgb datasets
    d_lgb_datasets = get_lgb_datasets(d_data)

    def objective(trial):

        if use_FL:
            fl = FocalLoss(
                gamma=trial.suggest_uniform("fl_gamma", 0, 10),
                alpha=trial.suggest_uniform("fl_alpha", 0, 1),
            )

        # # boosting_types = ["gbdt", "rf", "dart"]
        # boosting_types = ["gbdt", "dart"]
        # boosting_types = ["gbdt", "dart"]
        # boosting_type = trial.suggest_categorical("boosting_type", boosting_types)
        boosting_type = "gbdt"

        params = {
            "boosting": boosting_type,
            "n_jobs": cfg["n_jobs"],
            "objective": "binary",
            "first_metric_only": True,
            "verbosity": -1,
            "max_depth": trial.suggest_int("max_depth", 2, 63),
            "num_leaves": trial.suggest_int("num_leaves", 2, 4095),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-5, 10),
            "scale_pos_weight": trial.suggest_uniform("scale_pos_weight", 10.0, 30.0),
            "is_unbalance": False,  # due to scale_pos_weight being set
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
            "bagging_freq": 1,
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        }

        datasets = list(d_lgb_datasets.keys())
        s_dataset = trial.suggest_categorical("dataset", datasets)

        # Add a callback for pruning
        if cfg["optimize"] == "TPR":
            pruning_callback = LightGBMPruningCallback(trial, "TPR")
            feval = [lgb_eval_TPR_given_PPF]
            s_metric = "TPR-mean"

        elif cfg["optimize"] == "AUC":
            pruning_callback = LightGBMPruningCallback(trial, "auc")
            feval = None
            params["metric"] = "auc"
            s_metric = "auc-mean"

        elif cfg["optimize"] == "average_precision":
            pruning_callback = LightGBMPruningCallback(trial, "average_precision")
            feval = None
            params["metric"] = "average_precision"
            s_metric = "average_precision-mean"

        elif cfg["optimize"] == "focal_loss" and use_FL:
            pruning_callback = LightGBMPruningCallback(trial, "focal_loss")
            feval = [fl.lgb_eval]
            s_metric = "focal_loss-mean"

        else:
            raise AssertionError(f"{cfg['optimize']} not implemented yet.")

        print(params, s_dataset)

        N_iterations_max = 10_000
        early_stopping_rounds = 50
        fobj = fl.lgb_obj if use_FL else None
        # if boosting_type == "dart":
        #     N_iterations_max = 100
        #     early_stopping_rounds = None
        # if boosting_type == "rf":
        #     fobj = None

        cv_res = lgb.cv(
            params,
            d_lgb_datasets[s_dataset],
            num_boost_round=N_iterations_max,
            early_stopping_rounds=early_stopping_rounds,
            folds=d_data["cv_splits_validation"],
            # verbose_eval=True,
            verbose_eval=False,
            fobj=fobj,
            feval=feval,
            seed=42,
            callbacks=[pruning_callback],
        )

        num_boost_round = len(cv_res[s_metric])
        trial.set_user_attr("num_boost_round", num_boost_round)
        return cv_res[s_metric][-1]

    filename_optuna = f"./models/optuna__{name}.pkl"

    if Path(filename_optuna).is_file() and not cfg["force_HPO"]:
        study = joblib.load(filename_optuna)

    else:
        SEED = 42
        np.random.seed(SEED)

        direction = "minimize" if cfg["optimize"] in ["focal_loss"] else "maximize"

        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_warmup_steps=50),
        )

        study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=True)

        Path(filename_optuna).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, filename_optuna)

    print(f"Optimizing for {cfg['optimize']}")
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    d_optuna_all = dict(trial.params)
    d_optuna_all["num_boost_round"] = trial.user_attrs["num_boost_round"]

    print("  Params: ")
    for key_, value in d_optuna_all.items():
        print(f"    {key_}: {value}")

    if use_FL:
        model = FLModel(d_optuna_all, d_data).fit()

    else:
        params = params_from_optuna(d_optuna_all)
        dataset_train = dataset_train_from_optuna(d_optuna_all, d_data)

        model = lgb.train(
            params,
            dataset_train,
            num_boost_round=d_optuna_all["num_boost_round"],
            verbose_eval=100,
        )

    dicts["y_pred_proba"][key] = model.predict(d_data["X_test"])
    dicts["y_pred_proba_train"][key] = model.predict(d_data["X_train_val"])

    dicts["models"][key] = model

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)

    # return d_optuna_all, d_data


#%%


def get_df_results(dicts):

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

    if "ML__exclude_age" in df_results.index:
        index = [
            "ML",
            "LR",
            "ML__top_10",
            "LR__top_10",
            "ML__exclude_age",
            "only_age",
        ]
    else:
        index = [
            "ML",
            "LR",
            "ML__top_10",
            "LR__top_10",
            "only_age",
        ]

    df_results_save = df_results.loc[
        index,
        metrics,
    ]

    return df_results, df_results_save


def highlight_max(s):
    """
    highlight the maximum in a Series yellow.
    """
    is_max = s == s.max()
    return ["background-color: green" if v else "" for v in is_max]


def highlight_min(s):
    """
    highlight the maximum in a Series yellow.
    """
    is_max = s == s.min()
    return ["background-color: red" if v else "" for v in is_max]


def style_df_results(df):

    d_styling = {
        "TPR_mean": "{:,.2%}".format,
        "PPV_mean": "{:,.2%}".format,
        "ACC_mean": "{:,.2%}".format,
        "F1": "{:,.2%}".format,
        "MCC": "{:,.2%}".format,
        "ROC_AUC": "{:,.2%}".format,
        "PR_AUC": "{:,.2%}".format,
        "PRG_AUC": "{:,.2%}".format,
    }

    # fmt:off
    df_style = (df.style
        .apply(highlight_max, subset=list(d_styling.keys()))
        .apply(highlight_min, subset=list(d_styling.keys()))
        .format(d_styling)
    )
    # fmt:on

    return df_style

    #%%


def extract_data_ROC(dicts):

    data_ROC = {}

    for key in dicts["results"].keys():

        data_ROC[key] = {}

        data_ROC[key]["FPR"] = dicts["results"][key]["FPR_mean"]
        data_ROC[key]["TPR"] = dicts["results"][key]["TPR_mean"]
        data_ROC[key]["fpr"] = dicts["ROC"][key]["fpr"]
        data_ROC[key]["tpr"] = dicts["ROC"][key]["tpr"]

        data_ROC[key]["y_test"] = dicts["data"][key]["y_test"]
        data_ROC[key]["y_pred_proba"] = dicts["y_pred_proba"][key]

    return data_ROC


#%%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import PercentFormatter, MaxNLocator
import seaborn as sns
import matplotlib.patches as mpatches


def compute_suitable_area(d_ROC, PPF_cut_min, PPF_cut_max):

    key = "only_age"
    y_true = d_ROC[key]["y_test"]
    y_pred_proba = d_ROC[key]["y_pred_proba"]

    cutoff = compute_cutoff(
        y_true,
        y_pred_proba,
        N=100_000,
        PPF_cut=PPF_cut_min,
    )
    y_pred = y_pred_proba >= cutoff
    d_out_min = dict(_compute_performance_measures(y_true.values, y_pred))

    key = "ML"
    y_true = d_ROC[key]["y_test"]
    y_pred_proba = d_ROC[key]["y_pred_proba"]

    cutoff = compute_cutoff(
        y_true,
        y_pred_proba,
        N=100_000,
        PPF_cut=PPF_cut_max,
    )
    y_pred = y_pred_proba >= cutoff
    d_out_max = dict(_compute_performance_measures(y_true.values, y_pred))

    return d_out_min, d_out_max


def plot_ROC(
    fig,
    ax,
    d_ROC,
    PPF_cut_min,
    PPF_cut_max,
    keys,
    names,
    colors,
    markers,
):

    d_min, d_max = compute_suitable_area(
        d_ROC,
        PPF_cut_min=PPF_cut_min,
        PPF_cut_max=PPF_cut_max,
    )

    ax_inset = ax.inset_axes([0.53, 0.08, 0.4, 0.65])

    for i_model, key in enumerate(keys):

        # break

        kwargs_point = dict(
            label=names[i_model],
            color=colors[i_model],
            marker=markers[i_model],
            linestyle="None",  # for the legend
            markersize=8,
        )

        kwargs_line = dict(
            ls="-",
            color=colors[i_model],
            alpha=0.9,
            lw=1,
        )

        ax.plot(
            d_ROC[key]["FPR"],
            d_ROC[key]["TPR"],
            **kwargs_point,
        )
        ax.plot(
            d_ROC[key]["fpr"],
            d_ROC[key]["tpr"],
            **kwargs_line,
        )

        ax_inset.plot(
            d_ROC[key]["FPR"],
            d_ROC[key]["TPR"],
            **kwargs_point,
        )
        ax_inset.plot(
            d_ROC[key]["fpr"],
            d_ROC[key]["tpr"],
            **kwargs_line,
        )

    ax.set(
        xlabel=r"$\mathrm{False \,\, Positive \,\,  Rate}$",
        ylabel=r"$\mathrm{True \,\, Positive \,\, Rate}$",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.set_title("ROC", pad=10)

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_box_aspect(1)
    ax.tick_params(axis="x", pad=8)

    ax_inset.xaxis.set_major_locator(MaxNLocator(3))
    ax_inset.yaxis.set_major_locator(MaxNLocator(4))

    # fig.subplots_adjust(bottom=0.3, wspace=0.33)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)

    # for ax in [ax_inset, ax_PR_inset]:
    for item in (
        [ax_inset.title, ax_inset.xaxis.label, ax_inset.yaxis.label]
        + ax_inset.get_xticklabels()
        + ax_inset.get_yticklabels()
    ):
        item.set_fontsize(16)

    xlims = (d_min["FPR_mean"], d_max["FPR_mean"])
    ylims = (d_min["TPR_mean"], d_max["TPR_mean"])

    ax_inset.set(xlim=xlims, ylim=ylims)
    ax_inset.locator_params(axis="both", nbins=3)

    rectpatch, connects = ax.indicate_inset_zoom(
        ax_inset,
        ec="k",
        ls="-",
        label=None,
        # lw=1,
        # alpha=1,
    )

    kwargs_dashed_lines = dict(ls="--", color="k", alpha=0.4)

    ax.plot(
        [d_min["FPR_mean"], d_min["FPR_mean"]],
        [0, d_min["TPR_mean"]],
        **kwargs_dashed_lines,
    )
    ax.plot(
        [d_max["FPR_mean"], d_max["FPR_mean"]],
        [0, d_min["TPR_mean"]],
        **kwargs_dashed_lines,
    )
    ax.plot(
        [0, d_min["FPR_mean"]],
        [d_min["TPR_mean"], d_min["TPR_mean"]],
        **kwargs_dashed_lines,
    )
    ax.plot(
        [0, d_min["FPR_mean"]],
        [d_max["TPR_mean"], d_max["TPR_mean"]],
        **kwargs_dashed_lines,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.1, -0.07, 0.8, 0.07),
        bbox_transform=fig.transFigure,
        fancybox=False,
        shadow=False,
        ncol=len(keys),
        fontsize=20,
        handletextpad=0.0,
        mode="expand",
        # borderaxespad=0.0,
        # labelspacing=0.3,
        # columnspacing=1,
        frameon=True,
        edgecolor="black",
    )


def plot_risc_score(ax, d_risc_scores, risc_key):

    y_pred_proba = d_risc_scores[risc_key]["y_pred_proba"]
    y_test = d_risc_scores[risc_key]["y_test"]
    cutoff = d_risc_scores[risc_key]["cutoff"]

    xlim = (y_pred_proba.min(), y_pred_proba.max())

    kwargs = dict(
        bins=30,
        range=xlim,
        histtype="step",
        density=True,
        lw=2,
    )

    ax.hist(y_pred_proba[y_test == 1], label="Positive", **kwargs)
    ax.hist(y_pred_proba[y_test != 1], label="Negative", **kwargs)
    ax.axvline(cutoff, ls="--", color="k", label="Threshold")

    ax.legend()
    ax.set(xlabel="Risc Score", ylabel="Counts (normalised)", xlim=xlim)
    ax.locator_params(axis="x", nbins=4)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)


def make_risc_ROC_curve(
    d_risc_scores,
    d_ROC,
    PPF_cut_min,
    PPF_cut_max,
    include_ML__exclude_age=False,
):

    keys = [
        "ML",
        "LR",
        "ML__top_10",
        "LR__top_10",
        "ML__exclude_age",
        "only_age",
    ]

    names = [
        r"$\mathrm{ML33}$",
        r"$\mathrm{LR33}$",
        r"$\mathrm{ML10}$",
        r"$\mathrm{LR10}$",
        r"$\mathrm{ML}$-$\mathrm{NoAge}$",
        r"$\mathrm{Age}$",
    ]

    markers = [
        "s",
        "o",
        "s",
        "o",
        "s",
        "^",
    ]

    if not include_ML__exclude_age:
        keys.remove("ML__exclude_age")
        names.remove(r"$\mathrm{ML-NoAge}$")

        markers = [
            "s",
            "o",
            "s",
            "o",
            "^",
        ]

    # colors = ["#377eb8", "#e41a1c", "#84BDEB", "#f5abc9", "#048b00"]
    colors = [
        "#096B91",
        "#E8542E",
        "#0EBFC2",
        "#FF8D30",
        "#048B00",
        "#3EC106",
    ]

    fig, axes = plt.subplots(figsize=(15, 6), ncols=2, nrows=1)
    ax_risc, ax_ROC = axes

    risc_key = "ML"
    plot_risc_score(ax_risc, d_risc_scores, risc_key)
    plot_ROC(
        fig,
        ax_ROC,
        d_ROC,
        PPF_cut_min,
        PPF_cut_max,
        keys,
        names,
        colors,
        markers,
    )

    return fig


def make_ROC_curves(
    data_risc_scores,
    data_ROC,
    cfg_str,
    include_ML__exclude_age=False,
    cuts=None,
):

    # PPF_cut_min, PPF_cut_max = 0.125, 0.275

    if cuts is None:
        cuts = [(0.10, 0.30), (0.125, 0.275), (0.15, 0.25)]

    for y_label in data_risc_scores.keys():
        # break

        d_risc_scores = data_risc_scores[y_label]
        d_ROC = data_ROC[y_label]

        for PPF_cut_min, PPF_cut_max in cuts:

            # break

            fig_ROC = make_risc_ROC_curve(
                d_risc_scores,
                d_ROC,
                PPF_cut_min=PPF_cut_min,
                PPF_cut_max=PPF_cut_max,
                include_ML__exclude_age=include_ML__exclude_age,
            )

            figname = f"./figures/ROC__{y_label}__{cfg_str}__{PPF_cut_min:.3f}__{PPF_cut_max:.3f}.pdf"
            Path(figname).parent.mkdir(parents=True, exist_ok=True)

            figname_png = figname.replace(".pdf", ".png").replace("/ROC", "/pngs/ROC")
            Path(figname_png).parent.mkdir(parents=True, exist_ok=True)

            fig_ROC.savefig(figname, bbox_inches="tight")
            fig_ROC.savefig(figname_png, dpi=300, bbox_inches="tight")

            plt.close("all")


#%%


def get_shap_ordered_columns(dicts, use_FL, key="ML"):

    key = "ML"
    model = dicts["models"][key]
    X_test = dicts["data"][key]["X_test"]
    # explainer = shap.TreeExplainer(model.model if use_FL else model) # Explainer
    explainer = shap.Explainer(model.model if use_FL else model)
    shap_values = explainer(X_test)

    if len(shap_values.values.shape) == 3:
        df_shap_values = pd.DataFrame(
            shap_values.values[:, :, 1], columns=X_test.columns
        )
    else:
        df_shap_values = pd.DataFrame(shap_values.values, columns=X_test.columns)

    # fmt: off
    shap_ordered_columns = (df_shap_values
        .abs()
        .mean()
        .sort_values(ascending=False))
    # fmt: on

    return shap_ordered_columns


def get_ML_LR_shap_values(dicts, key, use_FL, use_test=False):

    model = dicts["models"][key]

    if "ML" in key:
        X_test = dicts["data"][key]["X_test"]
        X_train = dicts["data"][key]["X_train_val"]
        if use_test:
            X = X_test
        else:
            X = X_train
        # explainer = shap.TreeExplainer(model.model if use_FL else model)
        explainer = shap.Explainer(model.model if use_FL else model)

    elif "LR" in key:
        X_test_imputed_scaled = dicts["data"][key]["X_test_imputed_scaled"]
        X_train_val_imputed_scaled = dicts["data"][key]["X_train_val_imputed_scaled"]
        if use_test:
            X = X_test_imputed_scaled
        else:
            X = X_train_val_imputed_scaled
        explainer = shap.LinearExplainer(model, X)

    shap_values = explainer(X)

    if len(shap_values.values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def get_df_shap_top10(dicts, keys=("ML", "LR"), use_FL=False, use_test=False):

    shap_values_ML = get_ML_LR_shap_values(dicts, keys[0], use_FL, use_test=use_test)
    shap_values_LR = get_ML_LR_shap_values(dicts, keys[1], use_FL, use_test=use_test)

    data_ML = shap_values_ML.abs.mean(axis=0).values

    df_shap_values = pd.DataFrame(
        index=shap_values_ML.feature_names,
        data={
            "ML": data_ML,
            "LR": shap_values_LR.abs.mean(axis=0).values,
        },
    )

    df_shap_values["ML"] /= df_shap_values["ML"].sum()
    df_shap_values["LR"] /= df_shap_values["LR"].sum()

    df_shap_values = df_shap_values.sort_values("ML", ascending=False)

    ML_top10 = list(df_shap_values.sort_values("ML", ascending=False).index[:10])
    LR_top10 = list(df_shap_values.sort_values("LR", ascending=False).index[:10])

    top10_union = ML_top10 + LR_top10
    top10_union = list(dict.fromkeys(top10_union))

    everything_but_top10 = [
        col for col in shap_values_ML.feature_names if col not in top10_union
    ]

    df_shap_top10 = df_shap_values.loc[top10_union].sort_values("ML", ascending=False)

    df_shap_top10.loc[r"$\mathrm{Remaining \,\, variables}$"] = df_shap_values.loc[
        everything_but_top10
    ].sum(axis=0)

    # rename features to latex names:
    df_shap_top10 = df_shap_top10.rename(index=d_translate)
    shap_values_ML.feature_names = [
        d_translate[name] for name in shap_values_ML.feature_names
    ]
    shap_values_LR.feature_names = [
        d_translate[name] for name in shap_values_LR.feature_names
    ]

    return df_shap_top10.iloc[::-1], shap_values_ML, shap_values_LR


def extract_data_shap(
    dicts,
    use_FL=False,
    use_test=False,
):
    df_shap_top10, shap_values_ML, shap_values_LR = get_df_shap_top10(
        dicts,
        keys=("ML", "LR"),
        use_FL=use_FL,
        use_test=use_test,
    )
    data_shap = {
        "top10": df_shap_top10,
        "ML": shap_values_ML,
        "LR": shap_values_LR,
    }

    shap_models_top10 = get_df_shap_top10(
        dicts,
        keys=("ML__top_10", "LR__top_10"),
        use_FL=use_FL,
        use_test=use_test,
    )
    data_shap["top10__top_10"] = shap_models_top10[0]
    data_shap["ML__top_10"] = shap_models_top10[1]
    data_shap["LR__top_10"] = shap_models_top10[2]

    return data_shap


#%%

from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import PercentFormatter, MaxNLocator
import colorsys
import matplotlib


def scale_lightness(rgb, scale_l):
    if isinstance(rgb, str):
        rgb = matplotlib.colors.ColorConverter.to_rgb(rgb)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def autolabel(ax, rects, color, fontsize=18):
    """
    Attach a text label above each bar displaying its height
    """
    color = scale_lightness(color, scale_l=0.9)
    for rect in rects:
        # pass
        height = rect.get_height()
        width = rect.get_width()
        ax.text(
            x=width + 0.002,
            y=rect.get_y() + height / 2.0,
            s=r"$" + f"{width*100:.1f}" + r"\%" + r"$",
            ha="left",
            va="center",
            color=color,
            fontsize=fontsize,
        )


#%%


def shap_plot_global(
    ax,
    d_shap,
    colors,
    width=0.4,
    use_top10_model=False,
):

    if use_top10_model:
        key = "top10__top_10"
        legend_ML = r"$\mathrm{ML10}$"
        legend_LR = r"$\mathrm{LR10}$"
        d_shap[key] = d_shap[key].iloc[1:]
    else:
        key = "top10"
        legend_ML = r"$\mathrm{ML33}$"
        legend_LR = r"$\mathrm{LR33}$"

    ind = np.arange(len(d_shap[key]))

    rects1 = ax.barh(
        ind + width / 2,
        d_shap[key]["ML"],
        width,
        color=colors[0],
        label=legend_ML,
    )
    rects2 = ax.barh(
        ind - width / 2,
        d_shap[key]["LR"],
        width,
        color=colors[1],
        label=legend_LR,
    )

    # add some text for labels, title and axes ticks
    ax.set_xlabel(r"$\mathrm{Feature \,\, importance}$")
    ax.set_yticks(ind)
    ax.set_yticklabels(d_shap[key].index)
    # ax.set_title("Global", pad=15, fontsize=30)

    autolabel(ax, rects1, color=colors[0], fontsize=16)
    autolabel(ax, rects2, color=colors[1], fontsize=16)

    ax.xaxis.set_major_formatter(PercentFormatter(1))

    ax.set(
        xlim=(0, d_shap[key].max().max() + 0.045),
        ylim=(-1.5 * width, len(d_shap[key]) - 1.25 * width),
    )

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(22)

    ax.xaxis.label.set_fontsize(28)
    ax.xaxis.set_major_locator(MaxNLocator(4))

    # # Only show ticks on the left and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        fontsize=24,
    )


from copy import deepcopy


def make_shap_plots(
    data_shap,
    cfg_str,
    fontsize=16,
    suffix="",
):

    colors = ["#096B91", "#E8542E"]

    for y_label, d_shap in data_shap.items():
        # break

        fig, axes = plt.subplots(figsize=(18, 9), ncols=2)
        ax_global, ax_local = axes

        shap_plot_global(
            ax_global,
            d_shap,
            colors,
        )

        method = "ML"
        shap_values = d_shap[method]
        beeswarm(
            shap_values,
            max_display=11,
            ax=ax_local,
            fig=fig,
        )
        fig.tight_layout()

        filename = f"./figures/shap__{y_label}__{cfg_str}{suffix}"
        fig.savefig(filename + ".pdf", bbox_inches="tight", pad_inches=0.15)
        fig.savefig(filename.replace("figures/", "figures/pngs/") + ".png", dpi=200)

        # figs.append(fig)
        # plt.close("all")

    # return figs


#%%


from shap import Explanation
from shap.plots._labels import labels
from shap.plots._utils import convert_ordering, convert_color
from shap.plots import colors
from shap.utils import safe_isinstance


# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def beeswarm(
    shap_values,
    max_display=10,
    order=Explanation.abs.mean(0),
    color=None,
    axis_color="#333333",
    alpha=1,
    color_bar=True,
    ax=None,
    fig=None,
):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.
    Parameters
    ----------
    shap_values : Explanation
        This is an Explanation object containing a matrix of SHAP values (# samples x # features).
    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)
    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """

    color_bar_label = "Feature value"

    shap_exp = deepcopy(shap_values)
    base_values = shap_exp.base_values
    values = shap_exp.values
    features = shap_exp.data
    feature_names = shap_exp.feature_names

    order = convert_ordering(order, values)

    color = colors.red_blue
    color = convert_color(color)

    idx2cat = None
    num_features = values.shape[1]

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()
    feature_order = convert_ordering(order, Explanation(np.abs(values)))

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos, inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(
                feature_names[inds[max_ind]] + " + %d other features" % (len(inds) - 1)
            )
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum(
            [
                len(orig_inds[feature_order[i]])
                for i in range(num_features - 1, len(values[0]))
            ]
        )
        values[:, feature_order[num_features - 1]] = np.sum(
            [
                values[:, feature_order[i]]
                for i in range(num_features - 1, len(values[0]))
            ],
            0,
        )

    # build our y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]
    if num_features < len(values[0]):
        # yticklabels[-1] = "Sum of %d other features" % num_cut
        yticklabels[-1] = "Remaining variables"

    row_height = 0.4

    ax_was_None = True
    if ax is None or fig is None:
        fig, ax = plt.subplots(
            figsize=(8, min(len(feature_order), max_display) * row_height + 1.5)
        )
        ax_was_None = True

    ax.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]
        colored_feature = True
        fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(
            nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
        )
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if pos == 0:
            mask = (np.nanpercentile(shaps, 0.1) <= shaps) & (
                shaps <= np.nanpercentile(shaps, 99.9)
            )
            parts = ax.violinplot(
                shaps[mask],
                positions=[0],
                widths=[0.9],
                vert=False,
                showextrema=False,
                bw_method=0.1,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#777777")
                pc.set_edgecolor("black")
                pc.set_alpha(0.75)

        else:

            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            ax.scatter(
                shaps[nan_mask],
                pos + ys[nan_mask],
                color="#777777",
                vmin=vmin,
                vmax=vmax,
                s=16,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(
                shaps[np.invert(nan_mask)],
                pos + ys[np.invert(nan_mask)],
                cmap=color,
                vmin=vmin,
                vmax=vmax,
                s=16,
                c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

    # draw the color bar
    import matplotlib.cm as cm

    m = cm.ScalarMappable(cmap=color)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.set_ticklabels(["Low", "High"])

    cb.set_label(color_bar_label, size=18, labelpad=0)
    cb.ax.tick_params(labelsize=18, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    # cb.draw_all()

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.tick_params("y", labelsize=22, length=22, width=0.5, which="major")
    ax.tick_params("x", labelsize=22)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=26)
    ax.set_yticks(range(len(feature_inds)))
    ax.set_yticklabels(reversed(yticklabels), fontsize=22)
    ax.set_ylim(-1, len(feature_inds))

    if ax_was_None:
        return fig, ax


#%%

from datetime import datetime


def today_as_year():
    today = datetime.today()
    year = today.timetuple().tm_yday / 366 + today.year
    return year


#%%

# Age (70 years),
# female,
# knee replacement,
# use of walking tool,
# hemoglobin 120,
# height 160,
# weight 70 kg,
# plus hypertension,
# no familiarly VTE,
# with others,
# plus PSD,
# no cardiac disease,
# no diabetes,
#

# Køn: 0= female 1= male NB the calculated risk on the fictional patient is wrong!!!
# DM_type: 0= no diabetes, 1= Insulin treated 2= oral antidiabetics 3= diet treatment
# Led: 0 = hip 1= knee
# Civilstatus 0= with others 1= alone 2= institution


def get_patient(data_all):

    d_patient = {
        "sex": 0,  # female
        "civil_status": 0,  # with others
        "height": 160,  # height 160
        "weight": 70,  # weight 70 kg,
        "hb": 120,  # hemoglobin 120,
        "smoking": 0,  # ?
        "alcohol": 0,  # ?
        "walking_tool": 1,  # use of walking tool,
        "rested": 0,
        "snore": 0,
        "dm_type": 0,  # insulin-treated diabetes
        # "hypertension_yes_or_prescription": 1,  # plus hypertension,
        "hyper_colesterol": 0,
        "cardiac_disease": 0,  # no cardiac disease,
        "pulmonary_disease": 0,
        "cerebral_attack": 0,
        "previous_vte": 0,
        "family_vte": 0,
        "cancer": 0,
        "kidney": 0,
        # "psd": 1,  # plus PSD,
        "joint": 1,  # hip replacement,
        "age": 70,  # Age (70 years),
        # "potent_ak": 0,
        "year": 2017.5,
        "group_card": 0,
        "group_resp": 0,
        "group_psych": 0,
        "group_ak": 0,
        "steroid": 0,
        "cholesterol_medicine": 0,
        "antirheumatika": 0,
        "N_total_prescriptions": 0,
        "psd_knee": 0,
        "hypertens": 0,
    }

    d_patient["bmi"] = d_patient["weight"] / (d_patient["height"] / 100) ** 2

    X_patient = pd.DataFrame.from_dict(d_patient, orient="index").T

    columns = data_all["X"].columns

    X_patient = X_patient[columns]

    return X_patient


from collections import namedtuple


def get_shap_plot_object(shap_value, cutoff, max_display=10):

    base_value = shap_value.base_values
    if isinstance(base_value, float):
        pass
    else:
        if len(base_value) == 2:
            base_value = base_value[1]
    data_values = shap_value.data
    names = np.array(shap_value.feature_names)
    shaps = shap_value.values
    if len(shaps.shape) == 2:
        shaps = shaps[:, 1]

    N_display = min(max_display, len(shaps))
    N_rest = len(shaps) - N_display
    N_display_all = N_display + 2

    order_abs = np.argsort(-np.abs(shaps))

    shap_rest = shaps.sum() - shaps[order_abs][:N_display].sum()

    data_values = data_values[order_abs][:N_display]
    names = names[order_abs][:N_display]
    shaps = shaps[order_abs][:N_display]

    order_min2max = np.argsort(shaps)
    names = names[order_min2max]
    data_values = data_values[order_min2max]
    shaps = shaps[order_min2max]

    names = ["base value", f"{N_rest} other variables", *names]
    data_values = np.array([None, None, *data_values])
    shaps = np.array([base_value, shap_rest, *shaps])

    shaps_cumsum = np.array([0, *np.cumsum(shaps)[:-1]])

    data_hover_names = []
    for name, value in zip(names, data_values):
        if value is not None:
            if value.is_integer():
                value = int(value)
            else:
                value = round(value, 2)

            s = f"{name} = {value}"
        else:
            s = ""
        data_hover_names.append(s)

    cutoff_logodds = special.logit(cutoff)  # prob to log-odds
    # special.expit(0) # log-odds to prob

    ShapCollection = namedtuple(
        "ShapCollection",
        (
            "shaps "
            "shaps_cumsum "
            "names "
            "data_hover_names "
            "N_display_all "
            "cutoff "
            "cutoff_logodds "
        ),
    )

    return ShapCollection(
        shaps,
        shaps_cumsum,
        names,
        data_hover_names,
        N_display_all,
        cutoff,
        cutoff_logodds,
    )


def make_local_shap_plot(ax, shap_collection_patient, fontsize=16):

    # fig, axes = plt.subplots(figsize=(18, 10), ncols=2)
    # ax_global, ax = axes

    x = shap_collection_patient.shaps
    y_pos = np.arange(len(x))
    dx = shap_collection_patient.shaps_cumsum
    names = shap_collection_patient.names
    cutoff_logodds = shap_collection_patient.cutoff_logodds

    # colors = px.colors.qualitative.Set1
    blue = "#377eb8"  # colors[1]
    red = "#e41a1c"  # colors[0]

    colors = [blue if xi < 0 else red for xi in x]

    ax2 = ax.twiny()

    ax.barh(
        y=y_pos,
        width=x,
        left=dx,
        color=colors,
        align="center",
        height=1,
    )

    ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    xticks = ax.get_xticks()
    ax.set(xticks=xticks)

    xlim = ax.get_xlim()
    ylim = (-0.75, len(x) - 0.25)
    ax.set(
        yticks=y_pos,
        xlabel="Logit",
        xlim=xlim,
        ylim=ylim,
    )
    ax.set_yticklabels(names, fontsize=fontsize)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()  # labels read top-to-bottom

    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_label_position("bottom")

    def tick_function(X):
        return [f"{special.expit(x):.1%}" for x in X]

    ax2.set_xlim(xlim)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(tick_function(xticks), fontsize=fontsize)
    ax2.set_xlabel("Risc score", fontsize=fontsize)

    ax.axvline(cutoff_logodds, ls="--", color="k", alpha=0.3)

    ax.axvspan(xlim[0], cutoff_logodds, alpha=0.1, color=blue)
    ax.axvspan(cutoff_logodds, xlim[1], alpha=0.1, color=red)

    pred = (x + dx)[-1]
    ax.plot((pred, pred), (y_pos[-1], ylim[-1]), "-k")


#%%


def make_beeswarm_shap_plots(data_shap, cfg_str):

    for y_label, d_shap in data_shap.items():

        for method in ["ML", "LR"]:

            figname = f"./figures/feature_importance_beeswarm__{y_label}__{method}__{cfg_str}.pdf"

            # if len(d_shap[method].values.shape) == 3:
            # d_shap[method].values = d_shap[method].values[:, :, 1]

            shap.plots.beeswarm(d_shap[method], max_display=20, show=False)
            plt.savefig(figname, bbox_inches="tight")

            # figname_png = figname.replace(".pdf", ".png").replace(
            #     "/feature_importance", "/pngs/feature_importance"
            # )

            # plt.savefig(figname_png, bbox_inches="tight")
            plt.close("all")


#%%


def get_shap_patient(model, X_patient, use_FL):

    if use_FL:
        init_score = model.fl.init_score(model.d_data["y_train_val"])
        # offset = init_score   - special.logit(cutoff)

        explainer = shap.Explainer(model.model)
        shap_values = explainer(X_patient)
        shap_values.base_values += init_score

    else:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_patient)

    shap_patient = shap_values[0]

    return shap_patient


#%%


def compute_TPRs_given_PPF_cuts(y_test, y_pred_proba, PPF_cuts):

    TPRs = np.zeros(len(PPF_cuts))

    for i, PPF_cut in enumerate(PPF_cuts):
        cutoff, PPF_obs, TPR = compute_cutoff_PPF_TPR(
            y_test,
            y_pred_proba,
            PPF_cut,
        )
        TPRs[i] = TPR

    return TPRs


def compute_TPRs_all(data_risc_scores):

    delta = 0.001
    PPF_cuts = np.arange(0, 1 + delta, delta)

    d_TPRs = {}

    for y_label in data_risc_scores.keys():

        d_risc_scores = data_risc_scores[y_label]

        d_TPRs[y_label] = {}

        for model in ["ML", "LR", "ML__top_10"]:

            y_test = d_risc_scores[model]["y_test"]
            y_pred_proba = d_risc_scores[model]["y_pred_proba"]

            TPRs = compute_TPRs_given_PPF_cuts(
                y_test=y_test,
                y_pred_proba=y_pred_proba,
                PPF_cuts=PPF_cuts,
            )

            d_TPRs[y_label][model] = TPRs

    return d_TPRs, PPF_cuts


def plot_PPF_TPR(data_risc_scores, cfg_str):

    d_TPRs, PPF_cuts = compute_TPRs_all(data_risc_scores)

    for y_label, d_TPRs_y_label in d_TPRs.items():

        fig, ax = plt.subplots(figsize=(5, 5))
        for model, TPRs in d_TPRs_y_label.items():
            ax.plot(PPF_cuts, TPRs, "-", lw=2, label=model)

        ax.set(xlabel="PPF", ylabel="TPR", title=y_label, xlim=(0, 1), ylim=(0, 1))
        ax.legend()

        filename = f"./figures/PPF_TPR__{cfg_str}__{y_label}.pdf"
        fig.savefig(filename)


#%%


# Prescribed anticoagulants
d_group_ak = {
    0: "None",
    1: "VKA",
    2: "Heparin+ASA",
    3: "DOAC",
    4: "ASA",
    5: "Dipyradimol",
    6: "ADP",
    7: "ASA+Dipyradimol",
    8: "VKA+ASA",
    9: "DOAC+ASA",
    10: "VKA+ADP",
    11: "DOAC+ADP",
    12: "VKA+Heparin",
    # 13:
    14: "DOAC+ASA+ADP",
    15: "ASA+ADP",
    16: "ASA+ADP+Heparin",
    17: "ASA+ADP+Dipyradimol",
}


# Prescribed cardiac drugs
d_group_card = {
    0: "None",
    1: "diuretics",
    2: "angiotensin-II/ACE-inhib",
    3: "Ca^{2+}-antagonists",
    4: "betablockers",
    5: "nitrates",
    6: "other AHT",
    7: "other IHD-drugs",
    # 8: ,
    9: "2 antihypertensives",
    10: "β-blocker & 1 AHT",
    11: "3 AHT",
    12: "β-blocker & 2 AHT",
    13: "β-blocker & 3 AHT",
    14: "4 AHT",
    15: "β-blocker & 4 AHT",
    16: "other AHT & AHT",
    17: "nitrates & any AHT",
    18: "other IHD-drugs & any AHT/nitrate",
    19: "other antiarrythmics & any ATH",
}

# Psychotropics
d_group_psych = {
    0: "None",
    1: "SSRI/SNRI/NaRI",
    2: "other AD",
    3: "antipsychotics",
    4: "BZ",
    5: "anticholinergics/memantine",
    6: "anti-ADHD",
    7: "NaSSA",
    8: "other psychotropics",
    9: "SSRI & other AD",
    10: "SSRI & NaSSA",
    11: "SSRI & antipsychotics",
    12: "SSRI & other psychotropics",
    13: "BZ+ any psychotropic",
    14: "antipsychotics + any psychotropic",
    15: "anti-ADHD + any psychotropic",
    16: "NaSSA + any psychotropic",
    17: "other psychotropic & any specified psychotropic",
}

# Respiratory drugs
d_group_resp = {
    0: "None",
    1: "SABA",
    2: "LABA or LAMA",
    3: "inhalation steroid",
    4: "SABA & Ipratropium",
    5: "LABA & steroid",
    6: "LABA & LAMA & steroid",
    7: "LAMA & steroid",
    8: "LABA & LAMA",
    9: "other pulmonary drug",
    10: "other pulmonary drug & steroid",
    11: "SABA & LABA or LAMA",
    12: "SABA & LAMA or LABA & steroid",
}


d_labels = {
    "group_ak": d_group_ak,
    "group_card": d_group_card,
    "group_psych": d_group_psych,
    "group_resp": d_group_resp,
}
d_xaxis = {
    "group_ak": "Prescribed anticoagulants",
    "group_card": "Prescribed cardiac drugs",
    "group_psych": "Psychotropics",
    "group_resp": "Respiratory drugs",
}


def make_shap_scatter(shap_values, y_label, group, fignumber, y_range):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.text(
        0.98,
        0.98,
        fignumber,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=18,
    )
    ax.text(
        0.99,
        0.02,
        d_xaxis[group],
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=16,
    )

    shap_corrected = deepcopy(shap_values[:, d_translate[group]])
    d = {from_: to for to, from_ in enumerate(np.unique(shap_corrected.data))}
    shap_corrected.data = np.array([d[val] for val in shap_corrected.data])

    shap.plots.scatter(
        shap_values=shap_corrected,
        hist=False,
        color=shap_values[:, d_translate["age"]],
        ax=ax,
        x_jitter=0.8,
        alpha=0.8,
        **y_range[y_label],
    )
    plt.close("all")

    labels = list(d_labels[group].values())
    ticks = np.arange(len(labels))

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    # ax.tick_params(axis="x", rotation=45)
    # fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    # ax.set_xticklabels(labels, rotation=35, ha="right", position=(10, 0.01))
    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlabel("")
    ax.set_ylabel("SHAP value")

    return fig, ax
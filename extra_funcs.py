import numpy as np
import pandas as pd
from plotly.missing_ipywidgets import FigureWidget
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
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
filename = "Dtasætround3_NBI_Predict_PrimæreTXA_14_17_M ASTER_WORK_1_Hb.csv"

d_columns_rename = {
    "D_ODTO": "date",
    "Køn": "sex",
    "Civilstatus": "civil_status",
    "Height": "height",
    "Weight": "weight",
    "Hb": "hb",
    # "Hb_g_dL": "hb_g_dL",
    # "Anæmi": "anaemia",  # remove
    "Rygning": "smoking",
    "Alkohol": "alcohol",
    "Gangredskab": "walking_tool",
    "Udhvilet": "rested",
    "Snorken": "snore",
    "DM_type": "dm_type",  # diabetes type
    "Hypertension_ja_ell_recept": "hypertension_yes_or_prescription",
    "Hyperkolesterol": "hyper_colesterol",
    "Cardiac_disease": "cardiac_disease",
    "Pulmonary_disease": "pulmonary_disease",
    "Cerebral_attack": "cerebral_attack",
    "Tidl_VTE": "previous_vte",  # blodprop / årebetædndelse
    "Fam_VTE": "family_vte",
    # "AK_beh": "ak_beh",  # slettes
    "Cancer": "cancer",
    "Nyre": "kidney",  # nyresvigt
    "PsD": "psd",  # psykopharmaka
    "Led": "joint",
    "Alder": "age",
    "BMI": "bmi",
    # "recept_PsD": "prescription_psd",  # slettes
    "PotentAK": "potent_ak",  # blodfortyndende behandling, renset op, stærkeste medicin
    "Årstal": "year",
    "Hospital": "hospital",
    "Medical_outcome": "medical_outcome_old",  # more or less than 4 days in hospital (or re-indlæggelse)
    "Medical_outcome_inkl_obs_diag": "medical_outcome_new",  # more or less than 4 days in hospital (or re-indlæggelse)
    "Medicaloutcome3": "medical_outcome_3",
    "F_post_op_liggetid": "length_of_stay",
    "Risikoscore1": "risc_score1",
}


d_translate = {
    "sex": r"$\mathrm{Sex}$",
    "civil_status": r"$\mathrm{Civil \,\, status}$",
    "height": r"$\mathrm{Height}$",
    "weight": r"$\mathrm{Weight}$",
    "hb": r"$\mathrm{HB}$",
    # "anaemia": r"$\mathrm{Anemia}$",
    "smoking": r"$\mathrm{Smoking}$",
    "alcohol": r"$\mathrm{Alcohol}$",
    "walking_tool": r"$\mathrm{Walking \,\, tool}$",
    "rested": r"$\mathrm{Rested}$",
    "snore": r"$\mathrm{Snore}$",
    "dm_type": r"$\mathrm{DM}$",
    "hypertension_yes_or_prescription": r"$\mathrm{Hypertension}$",
    "hyper_colesterol": r"$\mathrm{Colesterol}$",
    "cardiac_disease": r"$\mathrm{Cardiac}$",
    "pulmonary_disease": r"$\mathrm{Pulmonary}$",
    "cerebral_attack": r"$\mathrm{Cerebral}$",
    "previous_vte": r"$\mathrm{Prev. \,\, VTE}$",
    "family_vte": r"$\mathrm{Fam. \,\, VTE}$",
    # "ak_beh": r"$\mathrm{AK}$",
    "cancer": r"$\mathrm{Cancer}$",
    "kidney": r"$\mathrm{Kidney}$",
    "psd": r"$\mathrm{PSD}$",
    "joint": r"$\mathrm{Joint}$",
    "age": r"$\mathrm{Age}$",
    "bmi": r"$\mathrm{BMI}$",
    # "prescription_psd": r"$\mathrm{PSD}$",
    "potent_ak": r"$\mathrm{AK \,\, (potent)}$",
    "year": r"$\mathrm{Year}$",
    "month": r"$\mathrm{Month}$",
    "day_of_week": r"$\mathrm{Week \,\, day}$",
    "day_of_month": r"$\mathrm{Day \,\, of \,\, month}$",
    "year_fraction": r"$\mathrm{Fractional \,\, year}$",
}


def add_date_info_to_df(df):
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_of_month"] = df["date"].dt.day
    df["year_fraction"] = df["date"].dt.day_of_year / 366 + df["year"]


def load_entire_dataframe(make_cuts=True):

    df = pd.read_csv(
        filename,
        sep=";",
        decimal=",",
        na_values=" ",
        parse_dates=["D_ODTO"],
    ).rename(columns=d_columns_rename)

    add_date_info_to_df(df)

    df["ID"] = df["Komb"].str[:8].astype(int)

    if make_cuts:
        df = df.query("30 <= weight <= 250 & 100 <= height <= 210").reset_index(
            drop=True
        )

    return df

def df_to_X(df):
    drop_columns = [
        "Komb",
        "date",
        "Anæmi",
        "AK_beh",
        "recept_PsD",
        "medical_outcome_new",
        "medical_outcome_3",
        "medical_outcome_old",
        "risc_score1",
        "cutoff6",
        "cutoff5",
        "ID",
        "length_of_stay",
    ]

    X = df.drop(columns=drop_columns)
    return X

def df_to_X_y(df, y_label):
    X = df_to_X(df)
    y = df.loc[:, y_label]
    return X, y


def remove_hospital_column(X):
    if "hospital" in X.columns:
        return X.drop(columns="hospital")
    else:
        return X


def remove_age_column(X):
    if "age" in X.columns:
        return X.drop(columns="age")
    else:
        return X


def get_numerical_and_categorical_features(X):

    numerical_features = [
        "height",
        "weight",
        "hb",
        "age",
        "bmi",
        "year",
        "month",
        "day_of_week",
        "day_of_month",
    ]
    categorical_features = [col for col in X.columns if col not in numerical_features]

    columns_cat_subset = X.nunique()[categorical_features]
    columns_cat_subset = columns_cat_subset[columns_cat_subset >= 3]
    columns_cat_subset = list(columns_cat_subset.index)

    return categorical_features, columns_cat_subset


#%%


#%%


def get_lgb_datasets(d_data):

    d_lgb_datasets = {}

    methods = ["simple", "hospital", "categorical_all", "categorical_subset"]
    if not "hospital" in d_data["X_train_val"].columns:
        methods.remove("hospital")

    for X, name in zip(
        [d_data["X_train_val"], d_data["X_train_val_imputed"]], ["", "imputed_"]
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

    elif "hospital" in method:
        dataset = lgb.Dataset(
            X,
            label=y,
            categorical_feature=["hospital"],
            free_raw_data=False,  # needed for categorical
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


# from imblearn.over_sampling import SMOTENC
# from imblearn.combine import SMOTETomek


# def resample(df, X_imputed, y, method, categorical_features=None):
#     """
#     First fit and resample the training set.
#     Afterwards fit and resample the training and validation set.
#     """

#     # only oversampling
#     if method == "SMOTENC":
#         if categorical_features is None:
#             raise AssertionError(f"categorical_features has to be set for SMOTENC")
#         resampler = SMOTENC(categorical_features=categorical_features, random_state=42)

#     # over and undersampling
#     elif method == "SMOTETomek":
#         resampler = SMOTETomek(random_state=42)

#     # X_resampled, y_resampled = resampler.fit_resample(X_imputed, y)

#     mask_train = df.year <= 2015
#     mask_val = df.year == 2016
#     mask_train_val = df.year <= 2016

#     X_resampled_train, y_resampled_train = resampler.fit_resample(
#         X_imputed[mask_train], y[mask_train]
#     )

#     X_resampled_train_val, y_resampled_train_val = resampler.fit_resample(
#         X_imputed[mask_train_val], y[mask_train_val]
#     )

#     return (
#         X_resampled_train,
#         y_resampled_train,
#         X_resampled_train_val,
#         y_resampled_train_val,
#     )


#%%


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

def plot_length_of_stay_sig_bkg(data_df, key="ML__exclude_hospital"):

    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)

    titles = [r"$\mathrm{medical \, \, outcome \,\, new}$", r"$\mathrm{medical \, \, outcome \,\, 3}$",]


    for i_ax, (y_label, df) in enumerate(data_df.items()):

        ax = axes[i_ax]

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
            label="Signal",
            **kwargs,
        )
        ax.hist(
            df_bkg.length_of_stay,
            label="Background",
            **kwargs,
        )

        ax.set(
            xlabel="Recovery Time [days]",
            ylabel="Normalized Counts",
            # xlim=(0 - 0.5, N_days - 0.5),
            xlim=(0, N_days),
        )
        ax.legend()
        ax.xaxis.get_major_locator().set_params(integer=True)

        ax.set_title(titles[i_ax], pad=10)

    return fig, ax


def extract_data_risc_scores(dicts):
    d_risc_scores = {}
    for key in dicts["data"].keys():
        d_risc_scores[key] = {
            'y_test': dicts["data"][key]["y_test"],
            'y_pred_proba': dicts["y_pred_proba"][key],
            'cutoff': dicts["results"][key]["cutoff"],
                     }
    return d_risc_scores


def plot_risc_score_distribution(data_risc_scores, key="ML__exclude_hospital"):


    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)

    titles = [r"$\mathrm{medical \, \, outcome \,\, new}$", r"$\mathrm{medical \, \, outcome \,\, 3}$",]


    for i_ax, (y_label, d_risc_scores) in enumerate(data_risc_scores.items()):
        # break

        ax = axes[i_ax]

        y_test = d_risc_scores[key]["y_test"]
        y_pred_proba = d_risc_scores[key]["y_pred_proba"]
        cutoff = d_risc_scores[key]["cutoff"]

        xlim = (y_pred_proba.min(), y_pred_proba.max())

        kwargs = dict(
            bins=30,
            range=xlim,
            histtype="step",
            density=True,
            lw=2,
        )

        ax.hist(y_pred_proba[y_test == 1], label="signal", **kwargs)
        ax.hist(y_pred_proba[y_test != 1], label="background", **kwargs)
        ax.axvline(cutoff, ls="--", color="k", label="Cutoff")

        ax.legend()
        ax.set(xlabel="Risc Score", ylabel="Normalized Counts", xlim=xlim)

        ax.set_title(titles[i_ax], pad=10)


    return fig, ax



def plot_nan_fractions(df, y):
    sig = df.loc[y == 1].isna().mean()
    bkg = df.loc[y == 0].isna().mean()
    df_nans = sig.to_frame(name="signal").join(bkg.to_frame(name="background"))
    df_nans = df_nans.query("(signal != 0) | (background != 0)")
    fig, ax = plt.subplots(figsize=(10, 8))
    df_nans.plot.barh(title="NAN fraction for signal and background", ax=ax)
    return fig, ax


def plot_hb_bmi_nans_sig_bkg(df, y):
    hb_sig = df[y == 1].hb
    hb_bkg = df[y == 0].hb
    hb_bmi_nan = df[df.bmi.isna()].hb
    hb_bmi_not_nan = df[~df.bmi.isna()].hb

    lw = 1.5

    fig, ax = plt.subplots(figsize=(10, 8))

    for x, label in [
        (hb_sig, "Signal"),
        (hb_bkg, "Background"),
        (hb_bmi_nan, "BMI NAN"),
        (hb_bmi_not_nan, "BMI not NAN"),
    ]:
        ax.hist(
            x,
            bins=100,
            range=(df.hb.min(), df.hb.max()),
            density=True,
            histtype="step",
            lw=lw,
            label=label,
        )

    ax.set(xlabel="hb", ylabel="Normalized Counts")
    ax.legend()

    return fig, ax


def plot_length_of_stay(df, y):

    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.hist(df['length_of_stay'], bins=10, range=(0, 10), histtype='step', density=True, label='All years')

    years = df.year.unique()
    for year in years:
        df_year = df.query(f"year == {year}")
        ax.hist(
            df_year["length_of_stay"],
            bins=10,
            range=(0, 10),
            histtype="step",
            density=True,
            lw=1.5,
            label=year,
        )

    ax.set(xlabel="Recovery Time [days]", ylabel="Normalized Counts")
    ax.legend()
    ax.set_yscale("log")

    return fig, ax


def plot_length_of_stay_bar(df, y, overflow_from=10):

    fig, ax = plt.subplots(figsize=(10, 8))

    years = df.year.unique()

    width = 1 / (len(years) * 1.5)  # the width of the bars

    for i, year in enumerate(years):

        df_year = df.query(f"year == {year}")

        counts_all = df_year["length_of_stay"].value_counts().sort_index()
        counts = counts_all.loc[:overflow_from]
        counts[overflow_from + 1] = counts_all.loc[overflow_from:].sum()

        x = counts.index + width * i + width / 2 - (len(years) * width) / 2

        ax.bar(x, counts, width, label=year)

    x_ticks = np.arange(overflow_from + 2)
    x_tick_labels = list(range(overflow_from + 1)) + [f"{overflow_from+1}+"]

    ax.set(
        xlabel="Recovery Time [days]",
        ylabel="Normalized Counts",
        yscale="log",
        xticks=x_ticks,
        xticklabels=x_tick_labels,
    )
    # ax.set_yscale("log")
    ax.legend()

    return fig, ax


def plot_length_of_stay_cumsum(df, y):

    fig, ax = plt.subplots(figsize=(10, 8))

    years = df.year.unique()
    for year in years:
        df_year = df.query(f"year == {year}")
        ax.hist(
            df_year["length_of_stay"],
            bins=10,
            range=(0, 10),
            histtype="step",
            density=True,
            cumulative=-1,
            lw=1.5,
            label=year,
        )

    ax.set(xlabel="Recovery Time [days]", ylabel="1-cumsum", title="Negative ")
    ax.legend()
    # ax.set_yscale("log")

    return fig, ax


def plot_monthly_counts(df, y_label):
    monthly_counts = df.resample("M", on="date").count()["date"]
    monthly_counts_signal = df.resample("M", on="date").sum()[y_label]
    fig, ax = plt.subplots(figsize=(10, 8))
    monthly_counts.plot(ax=ax, label="Total")
    monthly_counts_signal.plot(ax=ax, label="Signal")
    ax.set(ylabel="Monthly Counts", title="Counts pr. month", ylim=(0, None))
    ax.legend()
    return fig, ax


import plotly.express as px


def make_plotly_figure():
    df = load_entire_dataframe(make_cuts=False)
    fig = px.scatter(df, x="weight", y="height", color="age", hover_data=["sex"])
    fig.write_html("./figures/plotly.html")
    return fig


def plot_ML_score_vs_age(dicts):

    fig, ax = plt.subplots()

    key = "ML__without_hospital__without_age"

    mask = (dicts["data"][key]["y_test"] == 1).values

    ax.plot(
        dicts["data"]["only_age"]["X_test"]["age"].iloc[~mask],
        dicts["y_pred_proba"][key][~mask],
        ".",
        alpha=0.1,
    )

    ax.plot(
        dicts["data"]["only_age"]["X_test"]["age"].iloc[mask],
        dicts["y_pred_proba"][key][mask],
        ".",
        alpha=1,
    )

    ax.axhline(dicts["results"][key]["cutoff"], color="k", ls="--")


#%%


def plot_ROC_AUC(dicts):
    d_colors = {key: f"C{i}" for i, key in enumerate(dicts["y_pred_proba"].keys())}

    fig_ROC_PR, axes = plt.subplots(figsize=(14, 14), ncols=2, nrows=2)
    ax_ROC, ax_PR = axes[0, :]
    ax_ROC_zoom, ax_PR_zoom = axes[1, :]

    for key in dicts["y_pred_proba"].keys():

        ax_ROC.plot(
            dicts['ROC'][key]["fpr"],
            dicts['ROC'][key]["tpr"],
            "-",
            color=d_colors[key],
            alpha=0.2,
        )

        for ax in [ax_ROC, ax_ROC_zoom]:
            ax.errorbar(
                x=dicts["results"][key]["FPR_mean"],
                xerr=dicts["results"][key]["FPR_std"],
                y=dicts["results"][key]["TPR_mean"],
                yerr=dicts["results"][key]["TPR_std"],
                fmt=".",
                label=f"AUC = {dicts['results'][key]['ROC_AUC']:.3f}, {key}",
                color=d_colors[key],
            )

        ax_PR.plot(
            dicts["PR"][key]["recall"],
            dicts["PR"][key]["precision"],
            "-",
            color=d_colors[key],
            alpha=0.2,
        )
        for ax in [ax_PR, ax_PR_zoom]:
            ax.errorbar(
                x=dicts["results"][key]["TPR_mean"],
                xerr=dicts["results"][key]["TPR_std"],
                y=dicts["results"][key]["PPV_mean"],
                yerr=dicts["results"][key]["PPV_std"],
                fmt=".",
                label=key,
                color=d_colors[key],
            )

    for ax in [ax_ROC, ax_ROC_zoom]:
        ax.set(
            xlabel=r"$\Pr(\hat{y} \mid \neg y)$ (FPR)",
            ylabel=r"$\Pr(\hat{y} \mid y)$ (recall, TPR)",
            title="ROC curve",
        )

    for ax in [ax_ROC, ax_PR]:
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
        )

    for ax in [ax_PR, ax_PR_zoom]:
        ax.set(
            xlabel=r"$\Pr( \hat{y} \mid y )$ (recall, TPR)",
            ylabel=r"$\Pr( y \mid \hat{y} )$ (precision, PPV)",
            title="Precision Recall curve",
        )

    ax_ROC.legend(loc="lower right")
    ax_PR.legend(loc="upper right")
    plt.close("all")
    return fig_ROC_PR


#%%

from plotly.subplots import make_subplots


def plotly_ROC_PR_PRG(dicts, df_results, cfg):

    d_names = {
        "ML": {
            "name": "ML (all)",
            "color": 0,
            "dash": None,
            "symbol": "circle",
        },
        "LR": {
            "name": "LR (all)",
            "color": 0,
            "dash": "dash",
            "symbol": "square",
        },
        "ML__exclude_hospital": {
            "name": "ML (¬H)",
            "color": 1,
            "dash": None,
            "symbol": "circle",
        },
        "LR__exclude_hospital": {
            "name": "LR (¬H)",
            "color": 1,
            "dash": "dash",
            "symbol": "square",
        },
        "ML__top_10": {
            "name": "ML (top 10, ¬H)",
            "color": 2,
            "dash": None,
            "symbol": "circle",
        },
        "LR__top_10": {
            "name": "LR (top 10, ¬H)",
            "color": 2,
            "dash": "dash",
            "symbol": "square",
        },
        "ML__exclude_hospital__exclude_age": {
            "name": "ML (¬H, ¬A)",
            "color": 3,
            "dash": None,
            "symbol": "circle",
        },
        "LR__exclude_hospital__exclude_age": {
            "name": "LR (¬H, ¬A)",
            "color": 3,
            "dash": "dash",
            "symbol": "square",
        },
        "ML__exclude_age": {
            "name": "ML (¬A)",
            "color": 4,
            "dash": None,
            "symbol": "circle",
        },
        "LR__exclude_age": {
            "name": "LR (¬A)",
            "color": 4,
            "dash": "dash",
            "symbol": "square",
        },
        "only_age": {
            "name": "Only age",
            "color": 5,
            "dash": None,
            "symbol": "star",
        },
        "risc_score1": {
            "name": "risc_score1",
            "color": 6,
            "dash": None,
            "symbol": "hexagram",
        },
    }

    fig = make_subplots(rows=1, cols=3, subplot_titles=("ROC", "PR", "PRG"))

    colors = px.colors.qualitative.G10
    #

    marker_size = 10

    hovertemplate = (
        "Model: <b>%{text}</b> <br><br>"
        #
        "<b>Confusion Matrix</b>: <br>"
        "TP: %{customdata[0]:5d} <br>"
        "FP: %{customdata[1]:5d} <br>"
        "FN: %{customdata[2]:5d} <br>"
        "TN: %{customdata[3]:5d} <br><br>"
        #
        "<b>Metrics</b>: <br>"
        "TPR: %{customdata[4]:.1%} <br>"
        "PPV: %{customdata[5]:.1%} <br>"
        "FPR: %{customdata[6]:.1%} <br>"
        "F1:  %{customdata[7]:.1%} <br>"
        "ACC: %{customdata[8]:.1%} <br>"
        #
        "<b>AUCs</b>: <br>"
        "ROC: %{customdata[9]:.1%} <br>"
        "PR:  %{customdata[10]:.1%} <br>"
        "PRG: %{customdata[11]:.1%} <br><br>"
        #
        "<b>Other</b>: <br>"
        "RG:  %{customdata[12]:.1%} <br>"
        "PG:  %{customdata[13]:.1%} <br>"
        "PPF: %{customdata[14]:.1%} <br>"
        "cutoff:  %{customdata[15]:.3f} <br>"
        #
        "<extra></extra>"
    )

    hover_columns = [
        "TP",
        "FP",
        "FN",
        "TN",
        "TPR_mean",
        "PPV_mean",
        "FPR_mean",
        "F1",
        "ACC_mean",
        "ROC_AUC",
        "PR_AUC",
        "PRG_AUC",
        "RG",
        "PG",
        "PPF_mean",
        "cutoff",
    ]

    for key, kwargs in d_names.items():

        customdata = df_results.loc[key, hover_columns].values.reshape(1, -1)

        kwargs_line = dict(
            mode="lines",
            opacity=0.2,
            marker=dict(color=colors[kwargs["color"]]),
            legendgroup=kwargs["name"],
            showlegend=False,
            hoverinfo="skip",
            line_dash=kwargs["dash"],
        )
        kwargs_dots = dict(
            mode="markers",
            marker=dict(size=marker_size, color=colors[kwargs["color"]]),
            legendgroup=kwargs["name"],
            name=kwargs["name"],
            text=[kwargs["name"]],
            customdata=customdata,
            hovertemplate=hovertemplate,
            line_dash=kwargs["dash"],
            marker_symbol=kwargs["symbol"],
        )

        fig.add_scatter(
            x=[dicts["results"][key]["FPR_mean"]],
            y=[dicts["results"][key]["TPR_mean"]],
            **kwargs_dots,
            row=1,
            col=1,
        )

        fig.add_scatter(
            x=dicts['ROC'][key]["fpr"],
            y=dicts['ROC'][key]["tpr"],
            **kwargs_line,
            row=1,
            col=1,
        )

        fig.add_scatter(
            x=[dicts["results"][key]["TPR_mean"]],
            y=[dicts["results"][key]["PPV_mean"]],
            **kwargs_dots,
            showlegend=False,
            row=1,
            col=2,
        )

        fig.add_scatter(
            x=dicts["PR"][key]["recall"],
            y=dicts["PR"][key]["precision"],
            **kwargs_line,
            row=1,
            col=2,
        )

        if key == "risc_score1":
            continue

        fig.add_scatter(
            x=[dicts["results"][key]["RG"]],
            y=[dicts["results"][key]["PG"]],
            **kwargs_dots,
            showlegend=False,
            row=1,
            col=3,
        )

        fig.add_scatter(
            x=dicts["PRG"][key]["recall_gain"],
            y=dicts["PRG"][key]["precision_gain"],
            **kwargs_line,
            row=1,
            col=3,
        )

    fig.update_layout(
        title="ROC and PR curves",
        legend=dict(
            title_text="Models:",
            title_font_size=20,
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="right",
            x=1,
            font_size=12,
            bordercolor="grey",
            borderwidth=1,
            tracegroupgap=5,
        ),
        hoverlabel=dict(
            font_size=16,
            font_family="courier",
        ),
    )

    fig.update_xaxes(
        title_text=r"$\Pr(\hat{y} \mid \neg y) ,\ \text{(FPR)}$",
        range=[0, 1],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=r"$\Pr(\hat{y} \mid y) ,\ \text{(recall, TPR)}$",
        range=[0, 1],
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text=r"$\Pr( \hat{y} \mid y ) ,\ \text{(recall, TPR)} $",
        range=[0, 1],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text=r"$\Pr( y \mid \hat{y} ) ,\ \text{(precision, PPV)}$",
        range=[0, 1],
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text=r"$\text{Recall Gain}$", range=[0, 1], row=1, col=3)
    fig.update_yaxes(title_text=r"$\text{Precision Gain}$", range=[0, 1], row=1, col=3)

    fig.write_html(
        f"./figures/{cfg['optimize']}/plotly_ROC_PR.html", include_mathjax="cdn"
    )


#%%


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
def nb_compute_cutoff_PPF_TPR(
    y_true, y_pred_proba, minimum=-1, maximum=-1, N=100, PPF_cut=0.2
):

    if minimum == -1:
        minimum = np.nanmin(y_pred_proba)

    if maximum == -1:
        maximum = np.nanmax(y_pred_proba)

    cutoffs = np.linspace(minimum, maximum, N)
    for cutoff in cutoffs:
        y_pred = y_pred_proba >= cutoff
        d_cm = compute_confusion_matrix(y_true, y_pred)
        PPF = compute_PPF(d_cm)
        if PPF < PPF_cut:
            TPR = compute_TPR(d_cm)
            return cutoff, PPF, TPR

    return cutoff, PPF, np.nan


def compute_cutoff_PPF_TPR(
    y_true, y_pred_proba, minimum=-1, maximum=-1, N=100, PPF_cut=0.2
):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values
    return nb_compute_cutoff_PPF_TPR(y_true, y_pred_proba, minimum, maximum, N, PPF_cut)


def compute_cutoff(y_true, y_pred_proba, minimum=-1, maximum=-1, N=100, PPF_cut=0.2):
    return compute_cutoff_PPF_TPR(y_true, y_pred_proba, minimum, maximum, N, PPF_cut)[0]


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


def compute_performance_measures(dicts, key, cutoff=None, PPF_cut=0.2):

    y_true = dicts["data"][key]["y_test"]
    y_pred_proba = dicts["y_pred_proba"][key]

    if cutoff is None:
        cutoff = compute_cutoff(y_true, y_pred_proba, N=1000, PPF_cut=PPF_cut)

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
    dicts['ROC'][key] = {"fpr": fpr, "tpr": tpr, "threshold_roc": threshold_roc}

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


def get_data(y_label, exclude=None, include=None):

    if exclude is not None and include is not None:
        raise AssertionError(f"exclude and include cannot both be set.")

    # load data
    df = load_entire_dataframe()
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
    df_test = df.loc[mask_test]
    X_test = X.loc[mask_test]
    y_test = y.loc[mask_test]

    # train and validate on < 2017
    mask_train_val = df["date"] < "2017-01-01"
    df_train_val = df.loc[mask_train_val]
    X_train_val = X.loc[mask_train_val]
    y_train_val = y.loc[mask_train_val]

    # 6 validation time intervals of 2 months in 2016
    time_intervals_val = [f"2016-{i:02d}-01" for i in range(1, 12, 2)] + ["2017-01-01"]

    cv_splits_validation = get_train_test_splits(df_train_val, time_intervals_val)

    X_train_val_imputed, imputer = impute(X_train_val)
    X_test_imputed, _ = impute(X_test, imputer=imputer)

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

    return d_data


#%%

from sklearn.preprocessing import MinMaxScaler


def add_model_age_only(dicts, y_label, key, PPF_cut=0.2):

    d_data = get_data(y_label)
    X_test = d_data["X_test"]
    dicts["data"][key] = d_data

    scaler = MinMaxScaler()

    X_test_np = X_test["age"].values.reshape(-1, 1)
    dicts["y_pred_proba"][key] = scaler.fit_transform(X_test_np)[:, 0]

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)


#%%


def add_model_risc_score1(dicts, y_label, key, PPF_cut=0.2):
    d_data = get_data(y_label)
    dicts["data"][key] = d_data

    df_test = d_data["df_test"]
    dicts["y_pred_proba"][key] = df_test["risc_score1"] / df_test["risc_score1"].max()

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)


#%%

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def add_model_LR(dicts, y_label, key, exclude=None, include=None, PPF_cut=0.2):

    d_data = get_data(y_label, exclude=exclude, include=include)
    dicts["data"][key] = d_data

    clf = LogisticRegression(random_state=0, max_iter=10_000).fit(
        d_data["X_train_val_imputed"], d_data["y_train_val"]
    )

    dicts["y_pred_proba"][key] = clf.predict_proba(d_data["X_test_imputed"])[:, 1]
    dicts["y_pred_proba_train"][key] = clf.predict_proba(d_data["X_train_val_imputed"])[
        :, 1
    ]

    dicts["models"][key] = clf

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)


#%%

import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
import joblib


#%%

from copy import copy
from pathlib import Path


def get_train_test_splits(df, time_intervals):
    cv_splits = []
    for i, interval in enumerate(time_intervals[:-1]):
        idx_train = df.index[df["date"] < interval]
        mask_test = (interval <= df["date"]) & (df["date"] < time_intervals[i + 1])
        idx_test = df.index[mask_test]
        cv_splits.append((idx_train.values, idx_test.values))
    return cv_splits


#%%


def params_from_optuna(d_optuna_all):

    d_params = {}

    d_params_base = {
        "objective": "binary",
        "verbose": -1,
        "is_unbalance": False,
        "bagging_freq": 1,
    }

    d_params["optuna_all"] = {
        **d_params_base,
        **{
            "boosting": d_optuna_all["boosting_type"],
            "max_depth": d_optuna_all["max_depth"],
            "num_leaves": d_optuna_all["num_leaves"],
            "scale_pos_weight": d_optuna_all["scale_pos_weight"],
            "min_child_weight": d_optuna_all["min_child_weight"],
            "subsample": d_optuna_all["subsample"],
            "colsample_bytree": d_optuna_all["colsample_bytree"],
            "bagging_fraction": d_optuna_all["bagging_fraction"],
        },
    }

    params = d_params["optuna_all"]
    return params

def fl_from_optuna(d_optuna_all, d_data):

    y_train_val = d_data["y_train_val"]

    fl = FocalLoss(gamma=d_optuna_all["fl_gamma"], alpha=d_optuna_all["fl_alpha"])

    init_score = np.full_like(
        y_train_val, fl.init_score(y_train_val), dtype=float
    )
    fobj = fl.lgb_obj

    return fl, init_score, fobj

def dataset_train_from_optuna(d_optuna_all, d_data, init_score):

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
        dataset_train = dataset_train_from_optuna(self.d_optuna_all,
                                                  self.d_data,
                                                  self.init_score)
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
# %%
        return special.expit(
            self.fl.init_score(self.d_data["y_train_val"]) + self.model.predict(X)
        )

    def get_model(self):

        model = self.model
        alpha = self.d_optuna_all["fl_alpha"]
        gamma = self.d_optuna_all["fl_gamma"]
        y_train_val =self.d_data["y_train_val"]

        def predict_proba(self, X):
            fl = FocalLoss(gamma=gamma, alpha=alpha)
            pred = special.expit(fl.init_score(y_train_val) + model.predict(X))
            return np.array([1 - pred, pred]).T

        model.predict_proba = types.MethodType(predict_proba, model)
        return model

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
    exclude=None,
    include=None,
    PPF_cut=0.2,
):

    print(f"\n\nFitting ML model {key}. \n\n")

    cfg = copy(cfg)
    cfg["exclude"] = exclude

    d_data = get_data(y_label, exclude=exclude, include=include)
    dicts["data"][key] = d_data

    # define lgb datasets
    d_lgb_datasets = get_lgb_datasets(d_data)

    def objective(trial):

        fl = FocalLoss(
            gamma=trial.suggest_uniform("fl_gamma", 0, 10),
            alpha=trial.suggest_uniform("fl_alpha", 0, 1),
        )

        # boosting_types = ["gbdt", "rf", "dart"]
        boosting_types = ["gbdt", "dart"]
        boosting_type = trial.suggest_categorical("boosting_type", boosting_types)

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
            "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
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

        elif cfg["optimize"] == "focal_loss":
            pruning_callback = LightGBMPruningCallback(trial, "focal_loss")
            feval = [fl.lgb_eval]
            s_metric = "focal_loss-mean"

        else:
            raise AssertionError(f"{cfg['optimize']} not implemented yet.")

        print(params, s_dataset)

        N_iterations_max = 10_000
        early_stopping_rounds = 50
        fobj = fl.lgb_obj
        if boosting_type == "dart":
            N_iterations_max = 100
            early_stopping_rounds = None
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

    filename = f"./models/optuna_study__{y_label}__{name}.pkl"

    if Path(filename).is_file() and not cfg["force_HPO"]:
        study = joblib.load(filename)

    else:
        SEED = 42
        np.random.seed(SEED)

        # direction = "maximize" if cfg["optimize"] in ["TPR", "AUC"] else "minimize"
        direction = "minimize" if cfg["optimize"] in ["focal_loss"] else "maximize"

        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_warmup_steps=50),
        )

        study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=True)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, filename)

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


    model = FLModel(d_optuna_all, d_data).fit()

    dicts["y_pred_proba"][key] = model.predict(d_data["X_test"])
    dicts["y_pred_proba_train"][key] = model.predict(d_data["X_train_val"])

    dicts["models"][key] = model

    compute_performance_measures(dicts, key, PPF_cut=PPF_cut)

    # dicts["results"][key], dicts["y_pred"][key] = compute_performance_measures(
    #     d_data["y_test"], dicts["y_pred_proba"][key]
    # )

    # fpr, tpr, threshold_roc = roc_curve(d_data["y_test"], dicts["y_pred_proba"][key])
    # dicts['ROC'][key] = {"fpr": fpr, "tpr": tpr, "threshold_roc": threshold_roc}

    # precision, recall, threshold_pr = precision_recall_curve(
    #     d_data["y_test"], dicts["y_pred_proba"][key]
    # )
    # dicts["PR"][key] = {
    #     "precision": precision,
    #     "recall": recall,
    #     "threshold_pr": threshold_pr,
    # }

    # prg_curve = prg.create_prg_curve(
    #     d_data["y_test"].values, dicts["y_pred_proba"][key]
    # )
    # dicts["PRG"][key] = {
    #     "recall_gain": prg_curve["recall_gain"],
    #     "precision_gain": prg_curve["precision_gain"],
    # }

    return d_optuna_all, d_data


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

    df_results_save = df_results.loc[
        [
            "ML__exclude_hospital",
            "LR__exclude_hospital",
            "ML__top_10",
            "LR__top_10",
            "ML__exclude_hospital__exclude_age",
            "only_age",
        ],
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

    return df_style


#%%


def save_Troels_data(dicts, y_label):

    d_data = get_data(y_label, exclude="hospital")
    X_train = d_data["X_train_val_imputed"]
    y_train = d_data["y_train_val"]

    X_test = d_data["X_test_imputed"]
    y_test = d_data["y_test"]

    new_cols = [f"x{i}" for i, _ in enumerate(X_train.columns)]
    X_train.columns = new_cols
    X_test.columns = new_cols

    y_train = pd.DataFrame.from_dict(
        {
            "y": y_train,
            "y_ML": dicts["y_pred_proba_train"]["ML__exclude_hospital"],
            "y_LR": dicts["y_pred_proba_train"]["LR__exclude_hospital"],
        }
    )

    y_test = pd.DataFrame.from_dict(
        {
            "y": y_test,
            "y_ML": dicts["y_pred_proba"]["ML__exclude_hospital"],
            "y_LR": dicts["y_pred_proba"]["LR__exclude_hospital"],
        }
    )

    X_train.to_csv("./data_Troels/X_train.csv", index=False)
    X_test.to_csv("./data_Troels/X_test.csv", index=False)
    y_train.to_csv("./data_Troels/y_train.csv", index=False)
    y_test.to_csv("./data_Troels/y_test.csv", index=False)


#%%

def extract_data_ROC(dicts):

    data_ROC = {}

    for key in dicts["results"].keys():

        data_ROC[key] = {}

        data_ROC[key]["FPR"] = dicts["results"][key]["FPR_mean"]
        data_ROC[key]["TPR"] = dicts["results"][key]["TPR_mean"]
        data_ROC[key]["fpr"] = dicts['ROC'][key]["fpr"]
        data_ROC[key]["tpr"] = dicts['ROC'][key]["tpr"]

        data_ROC[key]["y_test"] = dicts["data"][key]["y_test"]
        data_ROC[key]["y_pred_proba"] = dicts["y_pred_proba"][key]

    return data_ROC

#%%


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import PercentFormatter, MaxNLocator
import seaborn as sns
import matplotlib.patches as mpatches


def compute_suitable_area(d_ROC, PPF_cut_min=0.15, PPF_cut_max=0.25):

    key = "ML__exclude_hospital"

    y_true = d_ROC[key]["y_test"]
    y_pred_proba = d_ROC[key]["y_pred_proba"]

    cutoff = compute_cutoff(
        y_true,
        y_pred_proba,
        N=1000,
        PPF_cut=PPF_cut_min,
    )
    y_pred = y_pred_proba >= cutoff
    d_out_min = dict(_compute_performance_measures(y_true.values, y_pred))

    cutoff = compute_cutoff(
        y_true,
        y_pred_proba,
        N=1000,
        PPF_cut=PPF_cut_max,
    )
    y_pred = y_pred_proba >= cutoff
    d_out_max = dict(_compute_performance_measures(y_true.values, y_pred))

    return d_out_min, d_out_max


def make_ROC_curve(data_ROC, PPF_cut_min, PPF_cut_max):

    fig, axes = plt.subplots(figsize=(13.5, 6), ncols=2, nrows=1)

    keys = [
        "ML__exclude_hospital",
        "LR__exclude_hospital",
        "ML__top_10",
        "LR__top_10",
        "ML__exclude_hospital__exclude_age",
        "only_age",
    ]


    titles = [r"$\mathrm{medical \, \, outcome \,\, new}$", r"$\mathrm{medical \, \, outcome \,\, 3}$",]

    names = [
        r"$\mathrm{ML}_\mathrm{All}$",
        r"$\mathrm{LR}_\mathrm{All}$",
        r"$\mathrm{ML}_\mathrm{10}$",
        r"$\mathrm{LR}_\mathrm{10}$",
        r"$\mathrm{ML}_{\neg \mathrm{Age}}$",
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
    # colors = ["#377eb8", "#e41a1c", "#84BDEB", "#f5abc9", "#048b00"]
    colors = [
        "#096B91",
        "#E8542E",
        "#0EBFC2",
        "#FF8D30",
        "#048B00",
        "#3EC106",
    ]


    for i_ax, d_ROC in enumerate(data_ROC.values()):
        # break

        d_min, d_max = compute_suitable_area(
            d_ROC,
            PPF_cut_min=PPF_cut_min,
            PPF_cut_max=PPF_cut_max,
        )

        ax = axes[i_ax]
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
            ylabel=r"$\mathrm{True \,\, Positive \,\, Rate}$" if i_ax==0 else None,
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.set_title(titles[i_ax], pad=10)

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_box_aspect(1)
        ax.tick_params(axis='x', pad=8)

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
        bbox_transform = fig.transFigure,
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

    return fig, ax


def make_ROC_curves(data_ROC, optimize):

    # PPF_cut_min, PPF_cut_max = 0.125, 0.275

    for PPF_cut_min, PPF_cut_max in [(0.1, 0.3), (0.125, 0.275), (0.15, 0.25)]:

        fig_ROC, ax_ROC = make_ROC_curve(
            data_ROC,
            PPF_cut_min=PPF_cut_min,
            PPF_cut_max=PPF_cut_max,
        )

        figname = f"./figures/ROC__{PPF_cut_min}__{PPF_cut_max}__{optimize}.pdf"
        Path(figname).parent.mkdir(parents=True, exist_ok=True)

        figname_png = figname.replace(".pdf", ".png").replace("/ROC", "/pngs/ROC")
        Path(figname_png).parent.mkdir(parents=True, exist_ok=True)

        fig_ROC.savefig(figname, bbox_inches="tight")
        fig_ROC.savefig(figname_png, dpi=300, bbox_inches="tight")

        plt.close("all")



#%%


def get_ML_LR_shap_values(dicts, key):

    model = dicts["models"][key]

    if "ML" in key:
        X_test = dicts["data"][key]["X_test"]
        X = X_test
        explainer = shap.TreeExplainer(model.model)

    elif "LR" in key:
        X_test_imputed = dicts["data"][key]["X_test_imputed"]
        X = X_test_imputed
        explainer = shap.LinearExplainer(dicts["models"][key], X)

    shap_values = explainer(X)
    return shap_values


def get_df_shap_top10(dicts):

    shap_values_ML = get_ML_LR_shap_values(dicts, "ML__exclude_hospital")
    shap_values_LR = get_ML_LR_shap_values(dicts, "LR__exclude_hospital")

    df_shap_values = pd.DataFrame(
        index=shap_values_ML.feature_names,
        data={
            "ML": shap_values_ML.abs.mean(axis=0).values,
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

    df_shap_top10.loc[r"$\mathrm{''Overflow''}$"] = df_shap_values.loc[
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


def extract_data_shap(dicts):
    df_shap_top10, shap_values_ML, shap_values_LR = get_df_shap_top10(dicts)
    data_shap = {'top10': df_shap_top10,
                 'ML': shap_values_ML,
                 'LR': shap_values_LR,
                 }
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


def autolabel(ax, rects, color, fontsize=20):
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
            s=r"$" + f"{width:.3f}" + r"$",
            ha="left",
            va="center",
            color=color,
            fontsize=fontsize,
        )


def make_feature_importance_curve(data_shap):

    colors = ["#377eb8", "#e41a1c"]
    colors = ["#096B91", "#E8542E"]

    titles = [r"$\mathrm{medical \, \, outcome \,\, new}$", r"$\mathrm{medical \, \, outcome \,\, 3}$",]

    width = 0.4

    fig, axes = plt.subplots(figsize=(18, 10), ncols=2)

    for i_ax, d_shap in enumerate(data_shap.values()):
        # break

        ax = axes[i_ax]

        ind = np.arange(len(d_shap['top10']))
        rects1 = ax.barh(
            ind + width / 2,
            d_shap['top10']["ML"],
            width,
            color=colors[0],
            label=r"$\mathrm{ML}_\mathrm{All}$",
        )
        rects2 = ax.barh(
            ind - width / 2,
            d_shap['top10']["LR"],
            width,
            color=colors[1],
            label=r"$\mathrm{LR}_\mathrm{All}$",
        )

        # add some text for labels, title and axes ticks
        ax.set_xlabel(r"$\mathrm{Feature \,\, importance}$")
        ax.set_yticks(ind)
        ax.set_yticklabels(d_shap['top10'].index)
        ax.set_title(titles[i_ax], pad=15, fontsize=30)

        autolabel(ax, rects1, color=colors[0], fontsize=18)
        autolabel(ax, rects2, color=colors[1], fontsize=18)

        # ax.xaxis.set_major_formatter(PercentFormatter(1))

        ax.set(
            xlim=(0, d_shap['top10'].max().max() + 0.045),
            ylim=(-1.5 * width, len(d_shap['top10']) - 1.25 * width),
        )

        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(24)

        ax.xaxis.label.set_fontsize(30)
        ax.xaxis.set_major_locator(MaxNLocator(4))

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.1),
        fontsize=30,
    )

    fig.tight_layout()


    return fig, ax

def make_feature_importance_curves(data_shap, optimize):

    fig, ax = make_feature_importance_curve(data_shap)

    figname = f"./figures/feature_importance__{optimize}.pdf"
    fig.savefig(figname, bbox_inches="tight")

    figname_png = figname.replace(".pdf", ".png").replace(
        "/feature_importance", "/pngs/feature_importance"
    )
    fig.savefig(figname_png, dpi=300, bbox_inches="tight")

    plt.close("all")


    for key, d_shap in data_shap.items():

        for method in ['ML', 'LR']:

            figname = f"./figures/feature_importance_beeswarm__{key}__{optimize}__{method}.pdf"

            shap.plots.beeswarm(d_shap[method], max_display=20, show=False)
            plt.savefig(figname, bbox_inches="tight")


            figname_png = figname.replace(".pdf", ".png").replace(
        "/feature_importance", "/pngs/feature_importance"
    )

            plt.savefig(figname_png, bbox_inches="tight")
            plt.close("all")



#%%

def to_int(x):
    if np.isnan(x):
        return x
    else:
        return int(x)


def print_latex_table_categoricals(df_cat):

    latex_table = (
        r"\begin{table}[]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\begin{tabular}{@{}ll@{}}"
        + "\n"
        + r"\toprule"
        + "\n\t"
        + r"Variable & Frequencies \\"
        + "\n\t"
        + r"\midrule"
        + "\n"
    )

    for col in df_cat.select_dtypes(include=np.number).columns:
        if col in ["ID", "year", "hospital", "month", "day_of_week"]:
            continue

        x = df_cat[col]
        nunique = x.nunique()
        order = np.sort(x.unique())
        unique = x.value_counts(normalize=True, dropna=False)[order] * 100
        s = ""
        for k, v in unique.items():
            s += f"{to_int(k)}: ${v:.1f}" + r"\%$, "

        if col == "hypertension_yes_or_prescription":
            col = "hypertension"

        latex_table += "\t" + r"\verb|" + col + r"| & " + s[:-2] + r" \\ " + "\n"

    latex_table += (
        r"\bottomrule" + "\n"
        r"\end{tabular}"
        + "\n"
        + r"\caption{}"
        + "\n"
        + r"\label{tab:my-table-categorical}"
        + "\n"
        + r"\end{table}"
        + "\n"
    )

    print(latex_table)

#%%

def print_latex_table_numericals(df_num):

    latex_table = (
        r"\begin{table}[]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\begin{tabular}{@{}lllllll@{}}"
        + "\n"
        + r"\toprule"
        + "\n\t"
        + r"Variable & min & mean & median & max & sigma & nans \\"
        + "\n\t"
        + r"\midrule"
        + "\n"
    )

    for col in df_num.select_dtypes(include=np.number).columns:
        if col in ["ID", "year", "hospital", "month", "day_of_week"]:
            continue

        x = df_num[col]
        s = (
            f"${np.min(x):.2f}$ & "
            f"${np.mean(x):.2f}$ & "
            f"${np.nanmedian(x):.2f}$ & "
            f"${np.max(x):.2f}$ & "
            f"${np.std(x):.2f}$ & "
            f"${np.mean(np.isnan(x))*100:.2f}" + r"\%$, "
        )

        latex_table += "\t" + r"\verb|" + col + r"| & " + s[:-2] + r" \\ " + "\n"

    latex_table += (
        r"\bottomrule" + "\n"
        r"\end{tabular}"
        + "\n"
        + r"\caption{}"
        + "\n"
        + r"\label{tab:my-table-numerical}"
        + "\n"
        + r"\end{table}"
        + "\n"
    )

    print(latex_table)


#%%

def make_numerical_column_histograms(df, savefig):
    df_num = df.loc[:, df.nunique() > 10]

    fig, axes = plt.subplots(figsize=(8, 8), ncols=3, nrows=3)
    axes_flat = axes.flatten()

    for i, col in enumerate(df_num.columns):
        x = df_num[col]
        ax = axes_flat[i]
        ax.hist(x, 30)
        ax.set(xlabel=col)
        ax.ticklabel_format(useOffset=False)

    fig.tight_layout()

    if savefig:

        figname = f"./figures/numerical_hists.pdf"
        Path(figname).parent.mkdir(parents=True, exist_ok=True)

        figname_png = figname.replace(".pdf", ".png").replace(
            "/numerical", "/pngs/numerical"
        )
        Path(figname_png).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(figname, bbox_inches="tight")
        fig.savefig(figname_png, dpi=300, bbox_inches="tight")

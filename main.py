import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
import extra_funcs
import shap
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from pathlib import Path
import joblib

# plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"text.usetex": False})

run_all_models = False
save_stuff = True
forced = True

y_label = "medical_outcome_new"
y_label = "medical_outcome_3"


#%%


cfg = dict()

# save_plots = False
if extra_funcs.is_hep():
    cfg["force_HPO"] = False
    cfg["n_trials"] = 1000
    cfg["n_jobs"] = 30
else:
    cfg["force_HPO"] = False
    cfg["n_trials"] = 1000
    cfg["n_jobs"] = 0

optimize = "AUC"
cfg["optimize"] = optimize  # "TPR", "focal_loss", "AUC", "average_precision"


file_ROC = Path("data_ROC.joblib")
file_shap = Path("data_shap.joblib")
file_df = Path("data_df.joblib")
file_data_all = Path("data_all.joblib")
file_risc_scores = Path("data_risc_scores.joblib")

if (
    not forced
    and file_ROC.exists()
    and file_shap.exists()
    and file_df.exists()
    and file_data_all.exists()
    and file_risc_scores.exists()
):
    data_ROC = joblib.load(file_ROC)
    data_shap = joblib.load(file_shap)
    data_df = joblib.load(file_df)
    data_all = joblib.load(file_data_all)
    data_risc_scores = joblib.load(file_risc_scores)

else:

    data_ROC = {}
    data_shap = {}
    data_df = {}
    data_risc_scores = {}

    for y_label in ["medical_outcome_new", "medical_outcome_3"]:

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

        # reload(extra_funcs)
        extra_funcs.add_model_age_only(dicts, y_label, key="only_age")
        extra_funcs.add_model_risc_score1(dicts, y_label, key="risc_score1")

        if run_all_models:
            extra_funcs.add_model_LR(
                dicts,
                y_label,
                key="LR",
            )

        extra_funcs.add_model_LR(
            dicts,
            y_label,
            key="LR__exclude_hospital",
            exclude="hospital",
        )

        if run_all_models:
            extra_funcs.add_model_LR(
                dicts,
                y_label,
                key="LR__exclude_age",
                exclude="age",
            )

        extra_funcs.add_model_LR(
            dicts,
            y_label,
            key="LR__exclude_hospital__exclude_age",
            exclude=["hospital", "age"],
        )

        #%%

        if run_all_models:
            extra_funcs.add_ML_model(
                cfg,
                dicts,
                y_label,
                key="ML",
                name=f"ML__{cfg['optimize']}__{cfg['n_trials']}",
            )

        d_optuna_all, d_data = extra_funcs.add_ML_model(
            cfg,
            dicts,
            y_label,
            key="ML__exclude_hospital",
            name=f"ML__exclude_hospital__{cfg['optimize']}__{cfg['n_trials']}",
            exclude="hospital",
        )

        if run_all_models:
            extra_funcs.add_ML_model(
                cfg,
                dicts,
                y_label,
                key="ML__exclude_age",
                name=f"ML__exclude_age__{cfg['optimize']}__{cfg['n_trials']}",
                exclude="age",
            )

        extra_funcs.add_ML_model(
            cfg,
            dicts,
            y_label,
            key="ML__exclude_hospital__exclude_age",
            name=f"ML__exclude_hospital__exclude_age__{cfg['optimize']}__{cfg['n_trials']}",
            exclude=["hospital", "age"],
        )

        #%%

        key = "ML__exclude_hospital"
        model = dicts["models"][key]
        X_test = dicts["data"][key]["X_test"]
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer(X_test)
        if len(shap_values.values.shape) == 3:
            df_shap_values = pd.DataFrame(
                shap_values.values[:, :, 0], columns=X_test.columns
            )
        else:
            df_shap_values = pd.DataFrame(shap_values.values, columns=X_test.columns)

        # fmt: off
        shap_ordered_columns = (df_shap_values
            .abs()
            .mean()
            .sort_values(ascending=False))
        # fmt: on

        extra_funcs.add_ML_model(
            cfg,
            dicts,
            y_label,
            key="ML__top_10",
            name=f"ML__top_10__{cfg['optimize']}__{cfg['n_trials']}",
            include=list(shap_ordered_columns.index[:10]),
        )

        extra_funcs.add_model_LR(
            dicts,
            y_label,
            key="LR__top_10",
            include=list(shap_ordered_columns.index[:10]),
        )

        # print(dicts['data']['ML__exclude_hospital']['X_train_val'].nunique()[shap_ordered_columns.index])

        data_ROC[y_label] = extra_funcs.extract_data_ROC(dicts)
        data_shap[y_label] = extra_funcs.extract_data_shap(dicts)
        data_df[y_label] = extra_funcs.extract_data_df(dicts)
        data_all = extra_funcs.extract_all_data_df(dicts)
        data_risc_scores[y_label] = extra_funcs.extract_data_risc_scores(dicts)

        if not save_stuff:
            continue

        df_results, df_results_save = extra_funcs.get_df_results(dicts)

        filename = f"./results/results__{y_label}__{optimize}__{cfg['n_trials']}.csv"
        df_results_save.to_csv(filename)

        df_style = extra_funcs.style_df_results(df_results_save)

    joblib.dump(data_ROC, file_ROC)
    joblib.dump(data_shap, file_shap)
    joblib.dump(data_df, file_df)
    joblib.dump(data_all, file_data_all)
    joblib.dump(data_risc_scores, file_risc_scores)

    print("\n\n\nfinished")

#%%

df = data_all["X_test"]
df_cat = df.loc[:, df.nunique() <= 10]
df_num = df.loc[:, df.nunique() > 10]


#%%


if False:
    extra_funcs.print_latex_table_categoricals(df_cat)
    extra_funcs.print_latex_table_numericals(df_num)

#%%

if False:
    extra_funcs.make_numerical_column_histograms(df, savefig=True)


#%%

x = x

#%%

from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from scipy import special

key = "ML__exclude_hospital"

model = dicts["models"][key]

X_test = dicts["data"][key]["X_test"]
y_test = dicts["data"][key]["y_test"]

X_train_val = dicts["data"][key]["X_train_val"]
y_train_val = dicts["data"][key]["y_train_val"]

y_pred_proba = dicts["y_pred_proba"][key]


shaps = data_shap["medical_outcome_3"]["ML"]
df_shap = pd.DataFrame(
    shaps.values,
    columns=list(X_train_val.columns),
)


# model.save_model("model.txt")

# model.dump_model()

# import joblib
# import dill

# joblib.dump(model, "model.joblib")

# with open("model.dill", "wb") as dill_file:
#     dill.dump(model, dill_file)


# model = joblib.load("model.joblib")


# type(RandomForestClassifier(n_estimators=50, max_depth=10))

# from sklearn.pipeline import Pipeline

# isinstance(RandomForestClassifier(n_estimators=50, max_depth=10), Pipeline)

# import types


# def predict_proba(self, X_test, y_train_val=y_train_val, gamma=6.40, alpha=0.78):
#     fl = extra_funcs.FocalLoss(gamma=gamma, alpha=alpha)
#     pred = special.expit(fl.init_score(y_train_val) + model.predict(X_test))
#     return np.array([1 - pred, pred]).T


# model.predict_proba = types.MethodType(predict_proba, model)

explainer = ClassifierExplainer(model.get_model(), X_test, y_test, shap="tree")
# # explainer.dump(f"explainer__{y_label}__{key}.joblib")

d_out = {
    "X_test": X_test,
    "y_test": y_test,
    "X_train_val": X_train_val,
    "y_train_val": y_train_val,
    "y_pred_proba": y_pred_proba,
    "df_shap": df_shap,
    "d_optuna_all": d_optuna_all,
    "d_data": d_data,
}

import pickle

with open(f"explainer__{y_label}__{key}.pkl", "wb") as pkl_file:
    pickle.dump(d_out, pkl_file)

data_shap


# joblib.dump(
#     d_out,
#     f"explainer__{y_label}__{key}.joblib",
# )

# ExplainerDashboard(explainer, simple=True).run()

# %%

if False:
    # if True:
    extra_funcs.make_ROC_curves(data_ROC, optimize)

    # if False:
    extra_funcs.make_feature_importance_curves(data_shap, optimize)

    # if False:
    fig_length_of_stay = extra_funcs.plot_length_of_stay_sig_bkg(data_df)[0]

    figname = f"./figures/LOS_hist__{optimize}.pdf"
    figname_png = figname.replace(".pdf", ".png").replace("/LOS", "/pngs/LOS")

    fig_length_of_stay.savefig(figname, bbox_inches="tight")
    fig_length_of_stay.savefig(figname_png, bbox_inches="tight", dpi=300)

    fig_risc_score_distribution = extra_funcs.plot_risc_score_distribution(
        data_risc_scores
    )[0]

    figname = f"./figures/risc_score_hist__{optimize}.pdf"
    figname_png = figname.replace(".pdf", ".png").replace(
        "/risc_score", "/pngs/risc_score"
    )
    fig_risc_score_distribution.savefig(figname, bbox_inches="tight")
    fig_risc_score_distribution.savefig(figname_png, bbox_inches="tight", dpi=300)


#%%


if False:

    for key in dicts["y_pred_proba"].keys():

        if key == "risc_score1":
            continue

        mask_sig = dicts["data"][key]["y_test"].values == 1
        mask_bkg = ~mask_sig

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            dicts["y_pred_proba"][key][mask_sig],
            100,
            label="Signal",
            histtype="step",
            density=True,
        )
        ax.hist(
            dicts["y_pred_proba"][key][mask_bkg],
            100,
            label="Background",
            histtype="step",
            density=True,
        )
        ax.axvline(dicts["results"][key]["cutoff"], ls="--", color="k", alpha=0.5)

        ax.legend()
        ax.set(title=key)


#%%

if not extra_funcs.is_hep() and False:
    extra_funcs.save_Troels_data(dicts, y_label)

#%%

if False:
    extra_funcs.plot_ML_score_vs_age(dicts)

#%%
if False:

    from optuna.visualization import plot_contour
    from optuna.visualization import plot_edf
    from optuna.visualization import plot_intermediate_values
    from optuna.visualization import plot_optimization_history
    from optuna.visualization import plot_parallel_coordinate
    from optuna.visualization import plot_param_importances
    from optuna.visualization import plot_slice

    plot_optimization_history(study)
    # plot_intermediate_values(study)
    plot_parallel_coordinate(study)
    plot_parallel_coordinate(study, params=["colsample_bytree", "dataset", "max_depth"])
    plot_parallel_coordinate(study, params=["colsample_bytree", "dataset", "max_depth"])

    # plot_contour(study)
    plot_contour(study, params=["colsample_bytree", "max_depth"])

    plot_slice(study)

    plot_param_importances(study)
    plot_edf(study)


#%%

# EDA figures

if False:
    fig_nan_fraction = extra_funcs.plot_nan_fractions(df, y)[0]
    fig_hb_bmi_nans_sig_bkg = extra_funcs.plot_hb_bmi_nans_sig_bkg(df, y)[0]
    fig_length_of_stay_bar = extra_funcs.plot_length_of_stay_bar(df, y)[0]
    fig_monthly_counts = extra_funcs.plot_monthly_counts(df)[0]

    extra_funcs.make_plotly_figure()

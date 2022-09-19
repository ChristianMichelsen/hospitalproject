#%%

from collections import defaultdict
from importlib import reload
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import extra_funcs

# plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"text.usetex": False})

# run_all_models = False
# run_all_models = True
# plot_stuff = False
plot_stuff = True
# save_stuff = False
save_stuff = True
forced = False
# forced = True
# add_ML_26 = True

use_FL = False
# use_FL = True
FL_str = "use_FL" if use_FL else "no_FL"

y_label = "outcome_B"
y_label = "outcome_A"
y_labels = ["outcome_A", "outcome_B"]
# y_labels = ["outcome_A"]  #

# PPF = 0.10
# PPF = 0.15
PPF = 0.20
# PPF = 0.25
# PPF = 0.30

do_calibration = True


# %%


# reload(extra_funcs)

cfg = dict()

# save_plots = False
if extra_funcs.is_hep():
    cfg["force_HPO"] = False
    cfg["n_trials"] = 1000
    cfg["n_jobs"] = 20
else:
    cfg["force_HPO"] = False
    cfg["n_trials"] = 1000
    cfg["n_jobs"] = 6

optimize = "AUC"  # "average_precision"
# optimize = "average_precision"  #
cfg["optimize"] = optimize  # "TPR", "focal_loss", "AUC", "average_precision"
cfg["FL_str"] = FL_str
cfg["PPF"] = PPF
# cfg["add_ML_26"] = add_ML_26

cfg_str = extra_funcs.cfg_to_str(cfg)
cfg_str_with_PPF = f"{cfg_str}__{PPF:.2f}"


file_ROC = Path(f"./data/data_ROC__{cfg_str_with_PPF}.joblib")
file_shap = Path(f"./data/data_shap__{cfg_str_with_PPF}.joblib")
file_df = Path(f"./data/data_df__{cfg_str_with_PPF}.joblib")
file_data_all = Path(f"./data/data_all__{cfg_str_with_PPF}.joblib")
file_risc_scores = Path(f"./data/data_risc_scores__{cfg_str_with_PPF}.joblib")
file_top_10_columns = Path(f"./data/top_10_columns__{cfg_str_with_PPF}.joblib")
file_models = Path(f"./data/models__{cfg_str_with_PPF}.joblib")
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./figures/pngs").mkdir(parents=True, exist_ok=True)

if (
    not forced
    and file_ROC.exists()
    and file_shap.exists()
    and file_df.exists()
    and file_data_all.exists()
    and file_risc_scores.exists()
    and file_top_10_columns.exists()
    and file_models.exists()
):
    data_ROC = joblib.load(file_ROC)
    data_shap = joblib.load(file_shap)
    data_df = joblib.load(file_df)
    data_all = joblib.load(file_data_all)
    data_risc_scores = joblib.load(file_risc_scores)
    top_10_columns = joblib.load(file_top_10_columns)
    models = joblib.load(file_models)

else:

    data_ROC = {}
    data_shap = {}
    data_df = {}
    data_risc_scores = {}
    top_10_columns = {}
    models = {}

    for y_label in y_labels:

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
            "models_uncalibrated",
            "y_pred_proba_uncalibrated",
            "y_pred_proba_train_uncalibrated",
        ]

        dicts = {}
        for k in dict_variables:
            dicts[k] = defaultdict()

        df_table = extra_funcs.get_table_df(y_label)
        if save_stuff:
            filename_table = f"./results/table__{y_label}.csv"
            Path(filename_table).parent.mkdir(parents=True, exist_ok=True)
            df_table.to_csv(filename_table)

        extra_funcs.add_model_age_only(
            dicts=dicts,
            y_label=y_label,
            key="only_age",
            PPF_cut=PPF,
        )

        extra_funcs.add_model_LR(
            dicts=dicts,
            y_label=y_label,
            key="LR",
            PPF_cut=PPF,
            do_calibration=do_calibration,
        )

        extra_funcs.add_model_LR(
            dicts=dicts,
            y_label=y_label,
            key="LR_median",
            PPF_cut=PPF,
            median_impute=True,
            do_calibration=do_calibration,
        )

        extra_funcs.add_ML_model(
            cfg=cfg,
            dicts=dicts,
            y_label=y_label,
            key=f"ML",
            use_FL=use_FL,
            name=f"{y_label}__ML__{cfg_str}",
            PPF_cut=PPF,
            do_calibration=do_calibration,
        )

        shap_ordered_columns = extra_funcs.get_shap_ordered_columns(
            dicts=dicts,
            use_FL=use_FL,
            key="ML",
            do_calibration=do_calibration,
        )

        extra_funcs.add_ML_model(
            cfg=cfg,
            dicts=dicts,
            y_label=y_label,
            use_FL=use_FL,
            key="ML__top_10",
            name=f"{y_label}__ML__top_10__{cfg_str}",
            include=list(shap_ordered_columns.index[:10]),
            PPF_cut=PPF,
            do_calibration=do_calibration,
        )

        extra_funcs.add_model_LR(
            dicts=dicts,
            y_label=y_label,
            key="LR__top_10",
            include=list(shap_ordered_columns.index[:10]),
            PPF_cut=PPF,
            do_calibration=do_calibration,
        )

        data_ROC[y_label] = extra_funcs.extract_data_ROC(dicts)
        data_shap[y_label] = extra_funcs.extract_data_shap(
            dicts=dicts,
            use_FL=use_FL,
            use_test=False,
            do_calibration=do_calibration,
        )
        data_df[y_label] = extra_funcs.extract_data_df(dicts)
        data_all = extra_funcs.extract_all_data_df(dicts)
        data_risc_scores[y_label] = extra_funcs.extract_data_risc_scores(dicts)
        top_10_columns[y_label] = shap_ordered_columns.index[:10]
        models[y_label] = dicts["models"]["ML"]

        df_results, df_results_save = extra_funcs.get_df_results(dicts)
        if save_stuff:
            filename_df_results = (
                f"./results/results__{y_label}__{cfg_str_with_PPF}.csv"
            )
            Path(filename_df_results).parent.mkdir(parents=True, exist_ok=True)
            df_results_save.to_csv(filename_df_results)

        df_style = extra_funcs.style_df_results(df_results_save)

    joblib.dump(data_ROC, file_ROC)
    joblib.dump(data_shap, file_shap)
    joblib.dump(data_df, file_df)
    joblib.dump(data_all, file_data_all)
    joblib.dump(data_risc_scores, file_risc_scores)
    joblib.dump(top_10_columns, file_top_10_columns)
    joblib.dump(models, file_models)

print("\n\n\n")
print("Finished loading models")
print("\n\n\n")


#%%


if plot_stuff:

    print("plotting stuff")
    # reload(extra_funcs)

    extra_funcs.make_ROC_curves(
        data_risc_scores,
        data_ROC,
        cfg_str_with_PPF,
        cuts=[(PPF - 0.05, PPF + 0.05)],
        plot_test=True,
        # include_ML__exclude_age=True,
        # add_ML_26=add_ML_26,
    )

    # extra_funcs.make_ROC_curves(
    #     data_risc_scores,
    #     data_ROC,
    #     cfg_str_with_PPF,
    #     cuts=[(PPF - 0.05, PPF + 0.05)],
    #     plot_test=False,
    #     # include_ML__exclude_age=True,
    #     # add_ML_26=add_ML_26,
    # )

    extra_funcs.make_shap_plots(
        data_shap,
        cfg_str,
        fontsize=18,
    )

# plt.close("all")

# A = data_all["df"].query("outcome_A == 1")
# B = data_all["df"].query("outcome_B == 1")

# print(len(A))
# print(len(B))

# set_A = set(A.index)
# set_B = set(B.index)

# set_A.difference(set_B)
# len(set_B.difference(set_A))


for y_label in y_labels:

    print(y_label)

    name = f"{y_label}__ML__{cfg_str}"
    filename_optuna = f"./models/optuna__{name}.pkl"

    study = joblib.load(filename_optuna)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    d_optuna_all = dict(trial.params)
    d_optuna_all["num_boost_round"] = trial.user_attrs["num_boost_round"]

    print("  Params: ")
    for key_, value in d_optuna_all.items():
        print(f"    {key_}: {value}")

    print("\n\n")


print("\n\n\nfinished")

# special.logit(cutoff) # prob to log-odds
# special.expit(0) # log-odds to prob


#%%

# reload(extra_funcs)

df_table = extra_funcs.get_table_df_train_test()
if save_stuff:
    filename_table = f"./results/table__train_test.csv"
    Path(filename_table).parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(filename_table)


#%%

#%%

x = x


#%%

# E = data_all['y'].sum()

E = 1476
n = 22017
phi = E / n
P = 33
MAPE = 0.05

B1 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
B1

B2 = np.exp((-0.508 + 0.259 * np.log(phi) + 0.504 * np.log(P) - np.log(MAPE)) / 0.544)
B2

ln_L_null = E * np.log(E / n) + (n - E) * np.log((n - E) / n)
max_R2_CS = 1 - np.exp(2 * ln_L_null / n)
max_R2_CS

frac_variability = 0.20
R2_anticipated = frac_variability * max_R2_CS
S = 0.9

B3 = P / ((S - 1) * np.log(1 - R2_anticipated / S))
B3

# %%
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

key = "ML"


y_test = data_risc_scores["outcome_A"][key]["y_test"]
y_pred_proba_calibrated = data_risc_scores["outcome_A"][key]["y_pred_proba"]


prob_true_calibrated, prob_pred_calibrated = calibration_curve(
    y_test,
    y_pred_proba_calibrated,
    n_bins=20,
)


print(brier_score_loss(y_test, y_pred_proba_calibrated))


y_pred_proba_uncalibrated = dicts["y_pred_proba_uncalibrated"][key]

prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(
    y_test,
    y_pred_proba_uncalibrated,
    n_bins=20,
)
print(brier_score_loss(y_test, y_pred_proba_uncalibrated))


#%%

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot([0, 1], [0, 1], linestyle="-")

ax.plot(
    prob_pred_uncalibrated,
    prob_true_uncalibrated,
    marker=".",
    label="uncalibrated",
)

if do_calibration:
    ax.plot(
        prob_pred_calibrated,
        prob_true_calibrated,
        marker=".",
        label="calibrated",
    )
ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")
ax.legend()

#%%


if False:

    from matplotlib.backends.backend_pdf import PdfPages

    d_translate = extra_funcs.d_translate

    y_label = "outcome_A"
    method = "ML"

    for y_label in y_labels:

        shap_values = data_shap[y_label][method]

        filename = f"./figures/shap_interation__{y_label}__{method}__ALL--age.pdf"
        with PdfPages(filename) as pdf:

            for key in d_translate.keys():

                fig, ax = plt.subplots(figsize=(10, 10))
                shap.plots.scatter(
                    shap_values=shap_values[:, d_translate[key]],
                    color=shap_values[:, d_translate["age"]],
                    ax=ax,
                    x_jitter=0.8,
                    alpha=0.5,
                )
                #
                # fig.savefig(filename, dpi=300)
                pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
                plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d["Title"] = "SHAP interaction values"
            d["Author"] = "Christian Michelsen"

#%%


if False:

    d_translate = extra_funcs.d_translate

    y_label = "outcome_A"
    method = "ML"

    y_range = {
        "outcome_A": {"ymin": -0.5, "ymax": 0.7},
        "outcome_B": {"ymin": -0.1, "ymax": 0.45},
    }

    fignumbers = [
        "3a)",
        "3b)",
        "3c)",
        "3d)",
    ]

    for y_label in y_labels:
        # break

        shap_values = data_shap[y_label][method]

        it = zip(filter(lambda s: "group" in s, d_translate.keys()), fignumbers)
        for group, fignumber in it:
            # break

            fig, ax = extra_funcs.make_shap_scatter(
                shap_values,
                y_label,
                group,
                fignumber,
                y_range,
            )

            filename = (
                f"./figures/shap_interation__{y_label}__{method}__{group}-age.pdf"
            )
            filename_png = filename.replace("figures/", "figures/pngs/").replace(
                ".pdf", ".png"
            )
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            fig.savefig(filename_png, dpi=300, bbox_inches="tight")


#%%


if False:

    d_risc_scores = data_risc_scores["outcome_A"]["ML"]
    cutoff = d_risc_scores["cutoff"]
    y_pred_proba = dicts["y_pred_proba"]["ML"]
    y_test = dicts["y_test"]["ML"]

    y_test.sum()

    (y_pred_proba > cutoff).sum()
    (y_pred_proba > 0.5).sum()
    (y_pred_proba > 0.6).sum()

    from sklearn.metrics import brier_score_loss

    brier_score_loss(y_test, y_pred_proba)
    brier_score_loss(y_test, y_pred_proba > cutoff)

    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=5)

    # plot perfectly calibrated

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(prob_pred, prob_true, marker=".")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")

    cutoff_LR = dicts["results"]["LR"]["cutoff"]
    y_pred_proba_LR = dicts["y_pred_proba"]["LR"]
    y_test = dicts["data"]["LR"]["y_test"]

    brier_score_loss(y_test, y_pred_proba_LR)

    prob_true_LR, prob_pred_LR = calibration_curve(y_test, y_pred_proba_LR, n_bins=10)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(prob_pred_LR, prob_true_LR, marker=".")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")


#%%

if False:
    X_patient = extra_funcs.get_patient(data_all)
    models["outcome_B"].predict(X_patient)
    X_patient2 = X_patient.copy()
    X_patient2["group_resp"] = 10.0
    models["outcome_B"].predict(X_patient2)

# %%

if False:

    d_risc_scores = data_risc_scores["outcome_A"]["ML"]
    cutoff = d_risc_scores["cutoff"]
    y_pred_proba = d_risc_scores["y_pred_proba"]

    worst = np.argmax(y_pred_proba)
    best = np.argmin(y_pred_proba)

    d_risc_scores["y_test"].iloc[worst]
    d_risc_scores["y_test"].iloc[best]

    X_worst = data_all["X_test"].iloc[worst]
    X_best = data_all["X_test"].iloc[best]

    y_pred_worst = models["outcome_A"].predict(X_worst)[0]
    y_pred_best = models["outcome_A"].predict(X_best)[0]

    X_worst_low_age = X_worst.copy()
    print(X_worst_low_age["age"])
    X_worst_low_age["age"] = 18
    X_best_high_age = X_best.copy()
    print(X_best_high_age["age"])
    X_best_high_age["age"] = 100

    y_pred_worst_low_age = models["outcome_A"].predict(X_worst_low_age)[0]
    y_pred_best_high_age = models["outcome_A"].predict(X_best_high_age)[0]

    print(f"Cutoff: {cutoff:.3f}")
    print(f"Worst patient: {y_pred_worst:.3f}, low age: {y_pred_worst_low_age:.3f}")
    print(f"Best patient: {y_pred_best:.3f}, high age: {y_pred_best_high_age:.3f}")
    # %%


#%%


if False:

    from copy import deepcopy

    d_translate = extra_funcs.d_translate

    shaps = data_shap["outcome_A"]["ML"]

    shaps_hb = shaps[:, d_translate["hb"]]
    shaps_age = shaps[:, d_translate["age"]]
    shaps_sex = shaps[:, d_translate["sex"]]
    shaps_joint = shaps[:, d_translate["joint"]]

    mask_women = shaps_sex.data == 0
    mask_men = shaps_sex.data == 1

    mask_knee = shaps_joint.data == 0
    mask_hip = shaps_joint.data == 1

    limits = {
        "ymin": np.min(shaps_hb.values),
        "ymax": np.max(shaps_hb.values),
        "xmin": np.nanmin(shaps_hb.data),
        "xmax": np.nanmax(shaps_hb.data),
    }

    fig_hb = extra_funcs.plot_shap_hb(shaps_hb, shaps_age, limits)
    filename = f"./figures/shap_interation__outcome_A__ML__hb.pdf"
    fig_hb.savefig(filename, dpi=300, bbox_inches="tight")

    shaps_hb_women = extra_funcs.get_masked_version(shaps_hb, mask_women)
    shaps_hb_men = extra_funcs.get_masked_version(shaps_hb, mask_men)
    shaps_age_women = extra_funcs.get_masked_version(shaps_age, mask_women)
    shaps_age_men = extra_funcs.get_masked_version(shaps_age, mask_men)

    fig_hb_gender = extra_funcs.plot_shap_hb_2_split(
        shaps_hb_women,
        shaps_hb_men,
        shaps_age_women,
        shaps_age_men,
        limits,
        "Women",
        "Men",
        ypos_text=0.85,
    )

    filename = f"./figures/shap_interation__outcome_A__ML__hb__gender.pdf"
    fig_hb_gender.savefig(filename, dpi=300, bbox_inches="tight")

    shaps_hb_hip = extra_funcs.get_masked_version(shaps_hb, mask_hip)
    shaps_hb_knee = extra_funcs.get_masked_version(shaps_hb, mask_knee)
    shaps_age_hip = extra_funcs.get_masked_version(shaps_age, mask_hip)
    shaps_age_knee = extra_funcs.get_masked_version(shaps_age, mask_knee)

    fig_hb_hip_knee = extra_funcs.plot_shap_hb_2_split(
        shaps_hb_hip,
        shaps_hb_knee,
        shaps_age_hip,
        shaps_age_knee,
        limits,
        "Hip",
        "Knee",
        ypos_text=0.85,
    )
    filename = f"./figures/shap_interation__outcome_A__ML__hb__hip-knee.pdf"
    fig_hb_hip_knee.savefig(filename, dpi=300, bbox_inches="tight")

    shaps_hb_women_hip = extra_funcs.get_masked_version(shaps_hb, mask_women & mask_hip)
    shaps_hb_men_hip = extra_funcs.get_masked_version(shaps_hb, mask_men & mask_hip)
    shaps_age_women_hip = extra_funcs.get_masked_version(
        shaps_age, mask_women & mask_hip
    )
    shaps_age_men_hip = extra_funcs.get_masked_version(shaps_age, mask_men & mask_hip)

    fig_hb_gender_hip = extra_funcs.plot_shap_hb_2_split(
        shaps_hb_women_hip,
        shaps_hb_men_hip,
        shaps_age_women_hip,
        shaps_age_men_hip,
        limits,
        "Women, hip",
        "Men, hip",
        ypos_text=0.85,
    )

    filename = f"./figures/shap_interation__outcome_A__ML__hb__gender__hip.pdf"
    fig_hb_gender_hip.savefig(filename, dpi=300, bbox_inches="tight")

    shaps_hb_women_knee = extra_funcs.get_masked_version(
        shaps_hb, mask_women & mask_knee
    )
    shaps_hb_men_knee = extra_funcs.get_masked_version(shaps_hb, mask_men & mask_knee)
    shaps_age_women_knee = extra_funcs.get_masked_version(
        shaps_age, mask_women & mask_knee
    )
    shaps_age_men_knee = extra_funcs.get_masked_version(shaps_age, mask_men & mask_knee)

    fig_hb_gender_knee = extra_funcs.plot_shap_hb_2_split(
        shaps_hb_women_knee,
        shaps_hb_men_knee,
        shaps_age_women_knee,
        shaps_age_men_knee,
        limits,
        "Women, knee",
        "Men, knee",
        ypos_text=0.85,
    )
    filename = f"./figures/shap_interation__outcome_A__ML__hb__gender__knee.pdf"
    fig_hb_gender_knee.savefig(filename, dpi=300, bbox_inches="tight")


#%%

if False:

    from copy import deepcopy

    d_translate = extra_funcs.d_translate

    shaps = data_shap["outcome_A"]["ML"]

    shaps_hb = shaps[:, d_translate["hb"]]
    shaps_age = shaps[:, d_translate["age"]]
    shaps_sex = shaps[:, d_translate["sex"]]
    shaps_joint = shaps[:, d_translate["joint"]]

    mask_women = shaps_sex.data == 0
    mask_men = shaps_sex.data == 1

    mask_knee = shaps_joint.data == 0
    mask_hip = shaps_joint.data == 1

    limits = {
        "ymin": np.min(shaps_age.values),
        "ymax": np.max(shaps_age.values),
        "xmin": np.nanmin(shaps_age.data),
        "xmax": np.nanmax(shaps_age.data),
    }

    reload(extra_funcs)
    fig_age = extra_funcs.plot_shap_hb(
        shaps_age,
        shaps_hb,
        limits,
        xpos_text=0.4,
        num_digigts=2,
        do_fix_colorbar_shape=False,
        variable="Age",
    )
    filename = f"./figures/shap_interation__outcome_A__ML__age.pdf"
    fig_age.savefig(filename, dpi=300, bbox_inches="tight")

    shaps_age_women = extra_funcs.get_masked_version(shaps_age, mask_women)
    shaps_age_men = extra_funcs.get_masked_version(shaps_age, mask_men)
    shaps_hb_women = extra_funcs.get_masked_version(shaps_hb, mask_women)
    shaps_hb_men = extra_funcs.get_masked_version(shaps_hb, mask_men)

    reload(extra_funcs)
    fig_age_gender = extra_funcs.plot_shap_hb_2_split(
        shaps_age_women,
        shaps_age_men,
        shaps_hb_women,
        shaps_hb_men,
        limits,
        "Women",
        "Men",
        xpos_text=0.4,
        num_digigts=2,
        do_fix_colorbar_shape=False,
        variable="Age",
    )

    filename = f"./figures/shap_interation__outcome_A__ML__age__gender.pdf"
    fig_age_gender.savefig(filename, dpi=300, bbox_inches="tight")

    shaps_age_hip = extra_funcs.get_masked_version(shaps_age, mask_hip)
    shaps_age_knee = extra_funcs.get_masked_version(shaps_age, mask_knee)
    shaps_hb_hip = extra_funcs.get_masked_version(shaps_hb, mask_hip)
    shaps_hb_knee = extra_funcs.get_masked_version(shaps_hb, mask_knee)

    fig_age_hip_knee = extra_funcs.plot_shap_hb_2_split(
        shaps_age_hip,
        shaps_age_knee,
        shaps_hb_hip,
        shaps_hb_knee,
        limits,
        "Hip",
        "Knee",
        xpos_text=0.4,
        num_digigts=2,
        do_fix_colorbar_shape=False,
        variable="Age",
    )
    filename = f"./figures/shap_interation__outcome_A__ML__age__hip-knee.pdf"
    fig_age_hip_knee.savefig(filename, dpi=300, bbox_inches="tight")

# %%

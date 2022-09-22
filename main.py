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
 
make_shape_plots = True
make_shape_plots = False

if make_shape_plots:

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


if make_shape_plots:

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

#%%


if make_shape_plots:

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

if make_shape_plots:

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

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

# run_all_models = False
run_all_models = True
save_stuff = True
# save_stuff = False
forced = False
# forced = True

use_FL = True
# use_FL = False
FL_str = "use_FL" if use_FL else "no_FL"

y_label = "outcome_A"
y_label = "outcome_B"
y_labels = ["outcome_A", "outcome_B"]

#%%

cfg = dict()

# save_plots = False
if extra_funcs.is_hep():
    cfg["force_HPO"] = False
    cfg["n_trials"] = 1000
    cfg["n_jobs"] = 30
else:
    cfg["force_HPO"] = False
    cfg["n_trials"] = 10
    cfg["n_jobs"] = 6

optimize = "AUC"  # "average_precision"
cfg["optimize"] = optimize  # "TPR", "focal_loss", "AUC", "average_precision"
cfg["FL_str"] = FL_str

cfg_str = extra_funcs.cfg_to_str(cfg)

file_ROC = Path(f"./data/data_ROC__{cfg_str}.joblib")
file_shap = Path(f"./data/data_shap__{cfg_str}.joblib")
file_df = Path(f"./data/data_df__{cfg_str}.joblib")
file_data_all = Path(f"./data/data_all__{cfg_str}.joblib")
file_risc_scores = Path(f"./data/data_risc_scores__{cfg_str}.joblib")
file_top_10_columns = Path(f"./data/top_10_columns__{cfg_str}.joblib")
file_models = Path(f"./data/models__{cfg_str}.joblib")
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

        # break

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

        df_table = extra_funcs.get_table_df(y_label)
        if save_stuff:
            filename_table = f"./results/table__{y_label}.csv"
            Path(filename_table).parent.mkdir(parents=True, exist_ok=True)
            df_table.to_csv(filename_table)

        # reload(extra_funcs)
        extra_funcs.add_model_age_only(dicts, y_label, key="only_age")
        extra_funcs.add_model_risc_score1(dicts, y_label, key="risc_score1")

        extra_funcs.add_model_LR(
            dicts,
            y_label,
            key="LR",
        )

        extra_funcs.add_model_LR(
            dicts,
            y_label,
            key="LR__exclude_age",
            exclude="age",
        )

        #%%

        extra_funcs.add_ML_model(
            cfg,
            dicts,
            y_label,
            key=f"ML",
            use_FL=use_FL,
            name=f"{y_label}__ML__{cfg_str}",
        )

        if run_all_models:
            extra_funcs.add_ML_model(
                cfg,
                dicts,
                y_label,
                use_FL=use_FL,
                key="ML__exclude_age",
                name=f"{y_label}__ML__exclude_age__{cfg_str}",
                exclude="age",
            )

        #%%

        shap_ordered_columns = extra_funcs.get_shap_ordered_columns(
            dicts,
            key="ML",
            use_FL=use_FL,
        )

        extra_funcs.add_ML_model(
            cfg,
            dicts,
            y_label,
            use_FL=use_FL,
            key="ML__top_10",
            name=f"{y_label}__ML__top_10__{cfg_str}",
            include=list(shap_ordered_columns.index[:10]),
        )

        extra_funcs.add_model_LR(
            dicts,
            y_label,
            key="LR__top_10",
            include=list(shap_ordered_columns.index[:10]),
        )

        data_ROC[y_label] = extra_funcs.extract_data_ROC(dicts)
        data_shap[y_label] = extra_funcs.extract_data_shap(
            dicts,
            use_FL=use_FL,
            use_test=False,
        )
        data_df[y_label] = extra_funcs.extract_data_df(dicts)
        data_all = extra_funcs.extract_all_data_df(dicts)
        data_risc_scores[y_label] = extra_funcs.extract_data_risc_scores(dicts)
        top_10_columns[y_label] = shap_ordered_columns.index[:10]
        models[y_label] = dicts["models"]["ML"]

        df_results, df_results_save = extra_funcs.get_df_results(dicts)
        if save_stuff:
            filename_df_results = f"./results/results__{y_label}__{cfg_str}.csv"
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


print("Finished loading models")


#%%

if save_stuff:

    print("Plotting stuff")

    reload(extra_funcs)
    extra_funcs.create_length_of_stay_sig_bkg(data_df)

    extra_funcs.make_ROC_curves(
        data_risc_scores,
        data_ROC,
        cfg_str,
        include_ML__exclude_age=True,
    )

    extra_funcs.make_beeswarm_shap_plots(data_shap, cfg_str)

    reload(extra_funcs)
    X_patient = extra_funcs.get_patient(data_all)

    extra_funcs.make_shap_plots(
        data_shap,
        data_risc_scores,
        models,
        X_patient,
        cfg_str,
        fontsize=18,
        use_FL=use_FL,
    )

    extra_funcs.make_shap_plots(
        data_shap,
        data_risc_scores,
        models,
        X_patient,
        cfg_str,
        fontsize=18,
        use_FL=use_FL,
        suffix="__with_walking_tool",
    )

    X_patient_no_walking = X_patient.copy()
    X_patient_no_walking["walking_tool"] = 0

    extra_funcs.make_shap_plots(
        data_shap,
        data_risc_scores,
        models,
        X_patient_no_walking,
        cfg_str,
        fontsize=18,
        use_FL=use_FL,
        suffix="__no_walking_tool",
    )

    X_patient_male = X_patient.copy()
    X_patient_male["sex"] = 1

    extra_funcs.make_shap_plots(
        data_shap,
        data_risc_scores,
        models,
        X_patient_male,
        cfg_str,
        fontsize=18,
        use_FL=use_FL,
        suffix="__male",
    )


A = data_all["df"].query("outcome_A == 1")
B = data_all["df"].query("outcome_B == 1")

print(len(A))
print(len(B))

set_A = set(A.index)
set_B = set(B.index)

set_A.difference(set_B)
len(set_B.difference(set_A))


for y_label in y_labels:

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


print("\n\n\nfinished")

# special.logit(cutoff) # prob to log-odds
# special.expit(0) # log-odds to prob

#%%

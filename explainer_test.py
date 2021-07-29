import joblib
import numpy as np
from scipy import special
import extra_funcs
import dashboard_utils
import datetime
import pickle
import pandas as pd
from importlib import reload

# reload(dashboard_utils)

#%%

with open("explainer__medical_outcome_3__ML__exclude_hospital.pkl", "rb") as pkl_file:
    d_in = pickle.load(pkl_file)

model, cutoff, columns = dashboard_utils.get_trained_model(d_in)

#%%

d_patient = {
    "sex": 1,  # female ?
    "civil_status": 1,  # living alone ?,
    "height": 160,  # height 160
    "weight": 80,  # weight 80 kg,
    "hb": 120,  # hemoglobin 120,
    "smoking": 0,  # ?
    "alcohol": 0,  # ?
    "walking_tool": 1,  # use of walking tool,
    "rested": 0,
    "snore": 0,
    "dm_type": 0,
    "hypertension_yes_or_prescription": 1,  # plus hypertension,
    "hyper_colesterol": 0,
    "cardiac_disease": 0,  # no cardiac disease,
    "pulmonary_disease": 0,
    "cerebral_attack": 0,
    "previous_vte": 0,
    "family_vte": 0,
    "cancer": 0,
    "kidney": 0,
    "psd": 1,  # plus PSD,
    "joint": 1,  # knee replacement,
    "age": 79,  # Age (79 years),
    "potent_ak": 0,
    "date": datetime.datetime.today(),
}


X_patient = dashboard_utils.dict_to_patient(d_patient, columns)
shap_patient = dashboard_utils.get_shap_patient(model, cutoff, X_patient)

#%%

shap_collection_patient = dashboard_utils.get_shap_plot_object(shap_patient)
fig = dashboard_utils.plot_shap_object(shap_collection_patient)
fig.show()


fig.write_html("shap_waterfall.html")

# %%

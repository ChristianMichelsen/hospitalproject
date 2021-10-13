import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from extra_funcs import (
    d_columns_rename,
    add_date_info_to_df,
    df_to_X_y,
    get_train_test_splits,
    get_data,
    load_entire_dataframe,
    get_lgb_datasets,
)

#%%


# y_label = "outcome_A"
# filename = "NBI_Predict_TXA_recepter.csv"
# filename = "Dtasætround3_NBI_Predict_PrimæreTXA_14_17_M ASTER_WORK_1_Hb.csv"
# exclude = None
# include = None
# init_score = None

# d_data = get_data(y_label, exclude=exclude, include=include, filename=filename)

# d_lgb_datasets = get_lgb_datasets(d_data)

# params = {
#     "n_jobs": 6,
#     "objective": "binary",
#     "num_leaves": 31,
#     "is_unbalance": True,
#     "metric": "auc",
# }

# s_metric = "auc-mean"

# s_dataset = "categorical_all"

# N_iterations_max = 10_000
# early_stopping_rounds = 50
# fobj = None
# feval = None


# cv_res = lgb.cv(
#     params,
#     d_lgb_datasets[s_dataset],
#     num_boost_round=N_iterations_max,
#     early_stopping_rounds=early_stopping_rounds,
#     folds=d_data["cv_splits_validation"],
#     verbose_eval=True,
#     fobj=fobj,
#     feval=feval,
#     seed=42,
# )

# num_boost_round = len(cv_res[s_metric])
# print(cv_res[s_metric][-1])

#%%


filename1 = "Dtasætround3_NBI_Predict_PrimæreTXA_14_17_M ASTER_WORK_1_Hb.csv"
df1 = load_entire_dataframe(filename=filename1)

filename2 = "NBI_Predict_TXA_recepter.csv"
df2 = load_entire_dataframe(filename=filename2)


df1_columns = set(df1.columns)
df2_columns = set(df2.columns)

print(df1_columns.difference(df2_columns))
print(df2_columns.difference(df1_columns))


# %%

# d_data1 = get_data("outcome_A", filename=filename1)
# d_data2 = get_data("outcome_A", filename=filename2)


# def get_Dataset(X, y):
#     return lgb.Dataset(X.values, label=y.values, feature_name=list(X.columns))


# param = {"num_leaves": 31, "objective": "binary", "is_unbalance": True}
# param["metric"] = "auc"
# num_round = 100


# train_data1 = get_Dataset(d_data1["X_train_val"], d_data1["y_train_val"])
# bst1 = lgb.train(param, train_data1, num_round)
# ypred1 = bst1.predict(d_data1["X_test"])


# train_data2 = get_Dataset(d_data2["X_train_val"], d_data2["y_train_val"])
# bst2 = lgb.train(param, train_data2, num_round)
# ypred2 = bst2.predict(d_data2["X_test"])


# columns_union = list(
#     set(d_data1["X_train_val"].columns).intersection(
#         set(d_data2["X_train_val"].columns)
#     )
# )

# train_dataA = get_Dataset(d_data1["X_train_val"][columns_union], d_data1["y_train_val"])
# bstA = lgb.train(param, train_dataA, num_round)
# ypredA = bstA.predict(d_data1["X_test"][columns_union])

# train_dataB = get_Dataset(d_data2["X_train_val"][columns_union], d_data2["y_train_val"])
# bstB = lgb.train(param, train_dataB, num_round)
# ypredB = bstB.predict(d_data2["X_test"][columns_union])

# #%%


# # %%

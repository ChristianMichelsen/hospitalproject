import pandas as pd

df = pd.read_csv(
    "Dtasætround2_NBI_Predict_PrimæreTXA_14_17_MASTER_WORK_1_Hb_Recepter_knee.csv",
    sep=";",
    decimal=",",
    na_values=" ",
)
df = df.query("30 <= Weight <= 250 & 100 <= Height <= 210")
dates = pd.to_datetime(df["D_ODTO"], format="%m/%d/%Y 0:00:00")
N_patients_in_test_set = ("2017-01-01" <= dates).sum()
print(N_patients_in_test_set)
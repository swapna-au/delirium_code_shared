# %%
# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SOURCE_PATH = Path("path/to/raw_data")
SAVE_PATH = Path("./output_dir")
SAVE_PATH.mkdir(exist_ok=True)

# %%
# Cell 2: Load the data
# Load the data
df = pd.read_excel(SOURCE_PATH / "EMR Path Hb from_jan_2023_.xlsx")


# %%
# Cell 3: Display the first few rows of the data
def display_data(data):
    print(data.head())


# %%
# Cell 4: Handle missing values
def handle_missing_values(data):
    data = data.fillna(method="ffill").fillna(method="bfill")
    return data


# Filter data
df = df[
    (df["Hospital"] == "EHS Box Hill")
    & (df["Admission Datetime"] >= "2023-01-01")
    & (df["Admission Datetime"] <= "2024-11-01")
]


# %%
# Cell 5:
def convert_result_val(row):
    """
    all the Result Val are strings
    check if the Result Val can be converted to a number, if yes, then return it
    as a number
    otherwise, if it can't or id it is a null value, then check the Normalcy column
    if the Normalcy column is "Normal", return 80
    if the Normalcy column is "Low", return 40
    """

    try:
        return float(row["Result Val"])
    except ValueError:
        if row["Normalcy"] == "Normal":
            return 80
        elif row["Normalcy"] == "Low":
            return 40
        else:
            return np.nan


df["Updated Result Val"] = df.apply(convert_result_val, axis=1)

# %%
# Cell 6:

# Group by Encounter ID and calculate mean
summary_df = (
    df.groupby("Encntr ID")
    .agg(
        Mean_Hb_Result_Val=("Updated Result Val", "mean"),
        lowest_Hb_Result_Val=("Updated Result Val", "min"),
        Visit_ID=("Visit ID", "first"),
        Encntr_Type=("Encntr Type", "first"),
        Encntr_Status=("Encntr Status", "first"),
        Sex=("Sex", "first"),
        Hospital=("Hospital", "first"),
        Ward=("Ward", "first"),
        Clinical_Unit=("Clinical Unit", "first"),
        Admission_Datetime=("Admission Datetime", "first"),
        Discharge_Datetime=("Discharge Datetime", "first"),
        Ur=("Ur", "first"),
        Unit=("Unit", "first"),
    )
    .reset_index()
)


# %%
# Cell 7:

# Check if result is positive or negative
summary_df["Critical_Hb_Status"] = summary_df["lowest_Hb_Result_Val"].apply(
    lambda x: "Critical" if x < 70 else "Non-Critical"
)

"""
Hb status = If the final Hb result val is < 130 g/litre and sex = male, 
then denote that as a positive Hb value, else negative; 
If Hb result val is < 120 g/litre and sex = female, 
then Positive or else negative
"""


def check_hb_status(row):
    if row["Sex"] == "M":
        if row["lowest_Hb_Result_Val"] < 130:
            return "Low"
        else:
            return "Normal"
    elif row["Sex"] == "F":
        if row["lowest_Hb_Result_Val"] < 120:
            return "Low"
        else:
            return "Normal"


summary_df["Hb_Status"] = summary_df.apply(check_hb_status, axis=1)


# %%
summary_df.to_csv(SAVE_PATH / "EMR_PathHb_Processed.csv", index=False)

# %%
# Check Cr Positive distribution
plt.figure(figsize=(8, 5))
sns.histplot(x="Hb_Status", data=summary_df)
plt.title("Positive Count")
plt.show()


# %%
# Cell 8: Run the main function with the file path

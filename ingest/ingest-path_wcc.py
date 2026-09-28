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
df = pd.read_excel(SOURCE_PATH / "EMR path wcc_____from_jan_2023.xlsx")


# %%
# Cell 3: Display the first few rows of the data
def display_data(data):
    print(data.head())


# %%
# Cell 4: Handle missing values
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
# Cell 6: checking the units for the resul all in x10*9/L
df["Unit"].unique()


# %%
"""
group by Encounter ID, loop through "result val" to get average of all results for that encounter ID in column 'Mean WCC result value'
add this numeric result in a separate column. Keep only single unique encounter ID on each row
include all other columns as identifiers of the encounter ID
include Age, sex, visit ID, ward, clinical unit, UR, unit.
new table should have unique encntr ID for each row with average result value
"""

# Group by Encounter ID and calculate mean
summary_df = (
    df.groupby("Encntr ID")
    .agg(
        Mean_WCC_Result_Val=("Updated Result Val", "mean"),
        Age=("Age", "first"),
        Sex=("Sex", "first"),
        Visit_ID=("Visit ID", "first"),
        Ward=("Ward", "first"),
        Clinical_Unit=("Clinical Unit", "first"),
        UR=("Ur", "first"),
        Unit=("Unit", "first"),
    )
    .reset_index()
)


# %%
# If the final WCC result val is  >  > 12000 cell/microlitre OR 12 10*9/L,
# then denote that as a positive WCC value, else negative

summary_df["WCC_Status"] = summary_df["Mean_WCC_Result_Val"].apply(
    lambda x: "Positive" if x > 12 else "Negative"
)

# %%
plt.figure(figsize=(8, 5))
sns.histplot(x="WCC_Status", data=summary_df)
plt.title("Positive Count")
plt.show()
# %%
summary_df.to_csv(SAVE_PATH / "EMR_Path_wcc_Processed.csv", index=False)
# %%

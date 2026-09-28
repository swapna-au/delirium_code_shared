"""
Recorded within 48 hours of the patient being admitted
"""

# %%
# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
from pathlib import Path

SOURCE_PATH = Path("path/to/raw_data")
SAVE_PATH = Path("./output_dir")
SAVE_PATH.mkdir(exist_ok=True)

# %%
# Cell 2: Load the data
df = pd.read_excel(SOURCE_PATH / "EMR risks_____from_jan_2023__.xlsx")

# %%
"""
filter for the same date range as the other dataframes
"""
df = df[
    (df["Clinical Event Performed Date Time"] >= "2023-01-01")
    & (df["Clinical Event Performed Date Time"] <= "2024-11-01")
]

# %%
"""
return the duplicate records
# """
# df[df.duplicated(
#     subset=["Encntr ID",
#             "Visit ID",
#             "Clinical Event ID",
#             "Clinical Event Performed Date Time",
#             "Clinical Event Code"], keep=False)]


# %%
# Cell 3: Drop Duplicates
df_deduplicated = df.drop_duplicates(
    subset=[
        "Encntr ID",
        "Visit ID",
        "Clinical Event ID",
        "Clinical Event Performed Date Time",
        "Clinical Event Code",
    ]
)


# %%
df_deduplicated_original = df_deduplicated.copy()

# %%
"""
write a for loop for each encounter ID, 
if there are multiple records for the same encounter ID,
then take the maximum value for 4at Score and BOC Risk Rank
for Nurse Verified Continence Risk, Nurse Verified Nutritional Risk, 
Nurse Verified Risk of Cogn Impairment, Nurse Verified Risk of Falls, 
Nurse Verified Risk of Infection, Nurse Verified Risk of Pressure Injury, 
if any record is "Yes" then "Yes" else "No"

you then want to have a cleaned dataframe with only one record per encounter ID
"""

for encounter_id in df_deduplicated["Encntr ID"].unique():
    df_encounter = df_deduplicated[df_deduplicated["Encntr ID"] == encounter_id]
    if len(df_encounter) > 1:
        """
        check if 4at score is in Clinical Event Key,
        if yes, then take filter for 4at score and take the maximum value
        otherwise set it to -1 to denote missing values
        similarly for BOC Risk Rank
        for Nurse Verified Continence Risk, Nurse Verified Nutritional Risk,
        Nurse Verified Risk of Cogn Impairment, Nurse Verified Risk of Falls,
        Nurse Verified Risk of Infection, Nurse Verified Risk of Pressure Injury
        filter and then "Yes" takes precedence over "No"
        """

        if "4at Score" in df_encounter["Clinical Event Key"].values:
            df_deduplicated.loc[
                df_deduplicated["Encntr ID"] == encounter_id, "4at Score"
            ] = df_encounter[df_encounter["Clinical Event Key"] == "4at Score"][
                "Clinical Event Result"
            ].max()  # take the maximum value
        else:
            df_deduplicated.loc[
                df_deduplicated["Encntr ID"] == encounter_id, "4at Score"
            ] = -1

        if "BOC Risk" in df_encounter["Clinical Event Key"].values:
            df_deduplicated.loc[
                df_deduplicated["Encntr ID"] == encounter_id, "BOC Risk"
            ] = df_encounter[df_encounter["Clinical Event Key"] == "BOC Risk"][
                "Clinical Event Result"
            ].max()
        else:
            df_deduplicated.loc[
                df_deduplicated["Encntr ID"] == encounter_id, "BOC Risk"
            ] = -1

        for risk in [
            "Nurse Verified Continence Risk",
            "Nurse Verified Nutritional Risk",
            "Nurse Verified Risk of Cogn Impairment",
            "Nurse Verified Risk of Falls",
            "Nurse Verified Risk of Infection",
            "Nurse Verified Risk of Pressure Injury",
        ]:
            if risk in df_encounter["Clinical Event Key"].values:
                df_deduplicated.loc[
                    df_deduplicated["Encntr ID"] == encounter_id, risk
                ] = (
                    "Yes"
                    if "Yes"
                    in df_encounter[df_encounter["Clinical Event Key"] == risk][
                        "Clinical Event Result"
                    ].values
                    else "No"
                )
            else:
                df_deduplicated.loc[
                    df_deduplicated["Encntr ID"] == encounter_id, risk
                ] = "No"

# %%
"""
drop the Clinical Event Key and Clinical Event Result columns
along with nan values
"""

df_emr_risks = df_deduplicated.drop(
    columns=["Clinical Event Key", "Clinical Event Result"]
).dropna()

df_emr_risks = df_emr_risks.drop_duplicates(
    subset=[
        "Encntr ID",
        "4at Score",
        "BOC Risk",
        "Nurse Verified Continence Risk",
        "Nurse Verified Nutritional Risk",
        "Nurse Verified Risk of Cogn Impairment",
        "Nurse Verified Risk of Falls",
        "Nurse Verified Risk of Infection",
        "Nurse Verified Risk of Pressure Injury",
    ]
)


# %%
# Cell 5:
def convert_boc_val(row):
    if row["BOC Risk"] == "Score = >2 High":
        return 3
    elif row["BOC Risk"] == "Score = 1-2 Mod":
        return 2
    elif row["BOC Risk"] == "Score = 0 Min":
        return 1
    else:
        return 0


df_emr_risks["BOC Risk Rank"] = df_emr_risks.apply(convert_boc_val, axis=1)
df_emr_risks["BOC Risk Rank"] = df_emr_risks["BOC Risk Rank"].astype(int)


# %%
# creating BOC Risk (ordinal) BOC Risk (nominal)
"""
BOC Risk (ordinal) logic:
if BOC Risk Rank is 0, then missing
if BOC Risk Rank is 1, then Low
if BOC Risk Rank is 2, then Moderate
if BOC Risk Rank is 3, then High

BOC Risk (nominal) logic:
if BOC Risk Rank is 0, then missing
if BOC Risk Rank is 1, then No
if BOC Risk Rank is 2 or 3, then Yes
"""

df_emr_risks["BOC Risk (ordinal)"] = df_emr_risks["BOC Risk Rank"].apply(
    lambda x: "Low" if x == 1 else "Moderate" if x == 2 else "High" if x == 3 else ""
)

df_emr_risks["BOC Risk (nominal)"] = df_emr_risks["BOC Risk Rank"].apply(
    lambda x: "No" if x == 1 else "Yes" if x > 1 else ""
)


# %%
"""
convert 4at score to numeric
0-3 : low
> 3 : high
-1 : missing
"""

df_emr_risks["4at Score Categorical"] = df_emr_risks["4at Score"].apply(
    lambda x: "Low" if int(x) < 4 and int(x) > -1 else "High" if int(x) > 3 else ""
)


# %%
# drop the BOC Risk Rank column

df_emr_risks = df_emr_risks.drop(columns=["BOC Risk Rank"])


# %%
df_emr_risks.to_csv(SAVE_PATH / "EMR_Risks_Processed.csv", index=False)
# %%

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
df = pd.read_csv(SOURCE_PATH / "speech_and_sensory___wcc_____from_jan_2023.csv")

# %%
"""
only interested in records with the following Clinical Event Key:
'AHA_hear aid',
'AHA_visual impairment',
'SD Hearing Impairment',
'SD Sensation/Touch Deficit',
'SD Speech Deficit',
'SD Visual Impairment',
"""

df_sensory = df[
    df["Clinical Event Key"].isin(
        [
            "AHA_hear aid",  # as long as there is a value, compare with SD Hearing impairment
            "AHA_visual impairment",
            "SD Hearing Impairment",
            # 'SD Sensation/Touch Deficit',
            "SD Speech Deficit",
            # 'SD Visual Impairment'
        ]
    )
]

# %%
"""
Filter the data for the same date range as the other dataframes
"""
df_sensory["Clinical Event Performed Date Time"] = pd.to_datetime(
    df_sensory["Clinical Event Performed Date Time"], format="mixed", dayfirst=True
)
_n_before_daterange = len(df_sensory)
df_sensory = df_sensory[
    (df_sensory["Clinical Event Performed Date Time"] >= "2023-01-01")
    & (df_sensory["Clinical Event Performed Date Time"] <= "2024-11-01")
]
print(f"Date-range filter: {_n_before_daterange - len(df_sensory)} rows excluded "
      f"(outside 2023-01-01 to 2024-11-01); {len(df_sensory)} retained.")

# %%
"""
Transform the Result Val column
if No = 0
and Yes = 1
"""

df_sensory["Result Val"] = df_sensory["Result Val"].apply(
    lambda x: 1 if x.strip() in ["Yes", "Left, Right", "Right", "Left"] else 0
)

# %%
# dropping duplicates
df_sensory = df_sensory.drop_duplicates(
    subset=["Encntr ID", "Visit ID", "Clinical Event ID"]
)

# %%
df_wider = df_sensory.pivot(
    index=[
        "Encntr ID",
        "Visit ID",
        "Clinical Event ID",
        "Clinical Event Performed Date Time",
    ],
    columns="Clinical Event Key",
    values="Result Val",
).reset_index()


# %%
def logical_or(row):
    a = row["AHA_hear aid"]
    b = row["SD Hearing Impairment"]

    if a == 1:
        return 1
    elif b == 1:
        return 1
    else:
        return 0


# %%
df_wider["is_hearing_impaired"] = df_wider.apply(lambda x: logical_or(x), axis=1)

# %%
# Group by 'Encntr ID' and apply logical OR operation for specified columns
columns_to_or = ["AHA_visual impairment", "SD Speech Deficit", "is_hearing_impaired"]


def collapse(group):
    return group.max()


# %%
df_wider = df_wider.fillna(-1)

# %%
summary_df = (
    df_wider.groupby(["Encntr ID", "Visit ID"])[columns_to_or]
    .apply(collapse)
    .reset_index()
)

# %%
summary_df.replace(-1, np.nan, inplace=True)

# %%
summary_df.to_csv(SAVE_PATH / "speech_and_sensory_processed.csv", index=False)
# %%

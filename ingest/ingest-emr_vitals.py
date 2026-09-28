"""
Recorded within 48 hours of the patient being admitted
"""

# %%
# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

SOURCE_PATH = Path("path/to/raw_data")
SAVE_PATH = Path("./output_dir")
SAVE_PATH.mkdir(exist_ok=True)

# %%
# Cell 2: Load the data
# df = pd.read_excel("EMR vitals from Jan 2023.xlsx")

# Vitals arrive as multiple .xlsx extracts (one per date range), in their own
# "vitals" subfolder of the raw data root.
source_folder = str(SOURCE_PATH / "vitals")

# List all files in the source folder
files = [f for f in os.listdir(source_folder) if f.endswith(".xlsx")]

# Initialize an empty list to hold dataframes
dfs = []


# %%
# Loop through each file and process it
for file in files:
    # Load the data
    df = pd.read_excel(os.path.join(source_folder, file), sheet_name="Query result")

    # Drop duplicates
    df = df.drop_duplicates(
        subset=[
            "Encntr ID",
            "Visit ID",
            "Clinical Event ID",
            "Event End Dt Tm Tz",
            "Clinical Event Date",
            "Clinical Event Time",
            "Clinical Event Code",
        ]
    )

    """
    Filter Clinical Event Key to only include the following:
    Family Worry
    New Change in Behaviour / Thinking
    Peripheral Pulse Rate
    Respiratory Rate
    Temperature Tympanic
    """
    df = df[
        df["Clinical Event Key"].isin(
            [
                "Family Worry",
                "New Change in Behaviour / Thinking",
                "Peripheral Pulse Rate",
                "Respiratory Rate",
                "Temperature Tympanic",
            ]
        )
    ]

    # print the sum of unique Enctr ID
    print(f"Unique Encntr ID in {file}: {df['Encntr ID'].nunique()}")

    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes into one
df = pd.concat(dfs, ignore_index=True)


# %%
# Drop duplicates
df = df.drop_duplicates(
    subset=[
        "Encntr ID",
        "Visit ID",
        "Clinical Event ID",
        "Event End Dt Tm Tz",
        "Clinical Event Date",
        "Clinical Event Time",
        "Clinical Event Code",
    ]
)

# %%
"""
Filter the data for the same date range as the other dataframes
"""
df = df[
    (df["Clinical Event Date"] >= "2023-01-01")
    & (df["Clinical Event Date"] <= "2024-11-01")
]


# %%
# pivot df to wider format using Clinical Event Key and Clinical Event Result as the value
df_wider = df.pivot(
    index=[
        "Encntr ID",
        "Visit ID",
        "Clinical Event ID",
        "Event End Dt Tm Tz",
        "Clinical Event Date",
        "Clinical Event Time",
        "Clinical Event Code",
    ],
    columns="Clinical Event Key",
    values="Clinical Event Results",
).reset_index()

df_wider = df_wider.where(pd.notna(df_wider), None)

# %%
"""
Run an apply function to convert all the follwing columns Peripheral Pulse Rate, Respiratory Rate, Temperature Tympanic
to numeric values, and if the value is not a number, convert it to -1
"""


def convert_to_numeric(row):
    try:
        return float(row)
    except:
        return -1


df_wider["Peripheral Pulse Rate"] = df_wider["Peripheral Pulse Rate"].apply(
    convert_to_numeric
)
df_wider["Respiratory Rate"] = df_wider["Respiratory Rate"].apply(convert_to_numeric)
df_wider["Temperature Tympanic"] = df_wider["Temperature Tympanic"].apply(
    convert_to_numeric
)

# %%
"""
Group by Encounter ID, for each of the following columns, 
get the maximum value and a mean value for each encounter ID
but for Peripheral Pulse Rate, Respiratory Rate, and Temperature Tympanic we
want to ignore the -1 values for both the max and mean calculations
"""

summary_df = (
    df_wider.groupby("Encntr ID")
    .agg(
        {
            "Peripheral Pulse Rate": [
                lambda x: x[x != -1].max(),
                lambda x: x[x != -1].mean(),
            ],
            "Respiratory Rate": [
                lambda x: x[x != -1].max(),
                lambda x: x[x != -1].mean(),
            ],
            "Temperature Tympanic": [
                lambda x: x[x != -1].max(),
                lambda x: x[x != -1].mean(),
            ],
            "Family Worry": lambda x: (
                "Yes" if "Yes" in x.values else "None" if x.isnull().all() else "No"
            ),
            "New Change in Behaviour / Thinking": lambda x: (
                "Yes" if "Yes" in x.values else "None" if x.isnull().all() else "No"
            ),
        }
    )
    .reset_index()
)

summary_df.columns = [
    "Encntr ID",
    "Peripheral Pulse Rate Max",
    "Peripheral Pulse Rate Mean",
    "Respiratory Rate Max",
    "Respiratory Rate Mean",
    "Temperature Tympanic Max",
    "Temperature Tympanic Mean",
    "Family Worry",
    "New Change in Behaviour / Thinking",
]

# %%
"""
Add the following columns:
If mean Peripheral Pulse rate > 90 bpm = High otherwise Normal, if no value then Nan
If mean respiratory rate > 20 br/min = High otherwise Normal, if no value then Nan
If Tympanic Temp > 38 or < 36 degrees = Positive if not in the range then Negative
and if no value then Nan
"""

# Add the additional columns based on the conditions provided
summary_df["Peripheral Pulse Rate Status"] = summary_df[
    "Peripheral Pulse Rate Mean"
].apply(lambda x: "High" if x > 90 else "Normal" if pd.notna(x) else np.nan)
summary_df["Respiratory Rate Status"] = summary_df["Respiratory Rate Mean"].apply(
    lambda x: "High" if x > 20 else "Normal" if pd.notna(x) else np.nan
)
summary_df["Temperature Tympanic Status"] = summary_df[
    "Temperature Tympanic Mean"
].apply(
    lambda x: "Positive" if x > 38 or x < 36 else "Neutral" if pd.notna(x) else np.nan
)

summary_df.columns = [
    "Encntr ID",
    "Peripheral Pulse Rate Max",
    "Peripheral Pulse Rate Mean",
    "Respiratory Rate Max",
    "Respiratory Rate Mean",
    "Temperature Tympanic Max",
    "Temperature Tympanic Mean",
    "Family Worry",
    "New Change in Behaviour / Thinking",
    "Peripheral Pulse Rate Status",
    "Respiratory Rate Status",
    "Temperature Tympanic Status",
]

# %%
# replace all 'None' values with np.nan
summary_df.replace("None", np.nan, inplace=True)


# %%
summary_df.to_csv(SAVE_PATH / "EMR_Vitals_Processed.csv", index=False)
# %%

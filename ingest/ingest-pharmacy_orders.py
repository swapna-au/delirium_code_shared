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

# Pharmacy orders arrive as multiple .xlsx extracts (one per date range), in
# their own "pharmacy" subfolder of the raw data root.
source_folder = str(SOURCE_PATH / "pharmacy")

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
    df = df.drop_duplicates(subset=["Encntr ID", "Visit ID", "Order ID", "Order Dt Tm"])

    # print the sum of unique Order ID
    print(f"Unique Order ID in {file}: {df['Order ID'].nunique()}")

    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes into one
df = pd.concat(dfs, ignore_index=True)


# %%
# Drop duplicates
df = df.drop_duplicates(subset=["Encntr ID", "Visit ID", "Order ID", "Order Dt Tm"])

# %%
"""
Filter the data for the same date range as the other dataframes
"""
df["Order Dt Tm"] = pd.to_datetime(df["Order Dt Tm"])
_n_before_daterange = len(df)
df = df[
    (df["Order Dt Tm"] >= "2023-01-01")
    & (df["Order Dt Tm"] <= "2024-11-01")
]
print(f"Date-range filter: {_n_before_daterange - len(df)} order rows excluded "
      f"(outside 2023-01-01 to 2024-11-01); {len(df)} retained.")

# %%
"""
Creating a drug dataframe with the score
"""
data = {
    "Drug Name": [
        "Alimemazine",
        "Alverine",
        "Alprazolam",
        "Aripiprazole",
        "Asenapine",
        "Atenolol",
        "Bupropion",
        "Captopril",
        "Cetirizine",
        "Chlorthalidone",
        "Cimetidine",
        "Clidinium",
        "Clorazepate",
        "Codeine",
        "Colchicine",
        "Desloratadine",
        "Diazepam",
        "Digoxin",
        "Dipyridamole",
        "Disopyramide",
        "Fentanyl",
        "Furosemide",
        "Fluvoxamine",
        "Haloperidol",
        "Hydralazine",
        "Hydrocortisone",
        "lloperidone",
        "Isosorbide",
        "Levocetirizine",
        "Loperamide",
        "Loratadine",
        "Metoprolol",
        "Morphine",
        "Nifedipine",
        "Paliperidone",
        "Prednisone",
        "Quinidine",
        "Ranitidine",
        "Risperidone",
        "Theophylline",
        "Trazodone",
        "Triamterene",
        "Venlafaxine",
        "Warfarin",
        "Amantadine",
        "Belladonna",
        "Carbamazepine",
        "Cyclobenzaprine",
        "Cyproheptadine",
        "Loxapine",
        "Meperidine",
        "Methotrimeprazine",
        "Molindone",
        "Nefopam",
        "Oxcarbazepine",
        "Pimozide",
        "Amitriptyline",
        "Amoxapine",
        "Atropine",
        "Benztropine",
        "Brompheniramine",
        "Carbinoxamine",
        "Chlorpheniramine",
        "Chlorpromazine",
        "Clemastine",
        "Clomipramine",
        "Clozapine",
        "Darifenacin",
        "Desipramine",
        "Dicyclomine",
        "Dimenhydrinate",
        "Diphenhydramine",
        "Doxepin",
        "Doxylamine",
        "Fesoterodine",
        "Flavoxate",
        "Hydroxyzine",
        "Hyoscyamine",
        "Imipramine",
        "Meclizine",
        "Methocarbamol",
        "Nortriptyline",
        "Olanzapine",
        "Orphenadrine",
        "Oxybutynin",
        "Paroxetine",
        "Perphenazine",
        "Promethazine",
        "Propantheline",
        "Propiverine",
        "Quetiapine",
        "Scopolamine",
        "Solifenacin",
        "Thioridazine",
        "Tolterodine",
        "Trifluoperazine",
        "Trihexyphenidyl",
        "Trimipramine",
        "Trospium",
    ],
    "ACB Score": (
        [1] * 44  # 44 drugs with ACB Score 1
        + [2] * 12  # 12 drugs with ACB Score 2
        + [3] * 43  # 43 drugs with ACB Score 3
    ),
}
# Create DataFrame
drug_df = pd.DataFrame(data)

# %%
drug_df["Drug Name"] = drug_df["Drug Name"].apply(lambda x: x.lower())

# %%
# Merge the two dataframes
df_combined = df.merge(
    drug_df,
    left_on="Order Catalogue Short Description",
    right_on="Drug Name",
    how="left",
)

# %%
"""
Drop the Drug Name column, and replace the NaN values in the ACB Score column with 0
"""

df_combined = df_combined.drop(columns=["Drug Name"])
df_combined["ACB Score"] = df_combined["ACB Score"].fillna(0)

# %%
"""
convert Order Dt Tm to datetime
"""
df_combined["Order Dt Tm"] = pd.to_datetime(df_combined["Order Dt Tm"])

# %%
"""
Group by Encounter ID and Visit ID 
 - and calculate the ACB Score : look a the unique values of the Order Catalogue Short Description
    and then sum the ACB score
 - count the number of unique Order Catalogue Short Description if ACB score is not 0
 - take the min of the Order Dt Tm
 - and the Order Status of the last order
"""

# summary_df = (
#     df_combined.groupby(["Encntr ID", "Visit ID"])
#     .agg(
#         Total_ACB_Score=("ACB Score", "max"),
#         Unique_Drug_Count=(
#             "Order Catalogue Short Description",
#             lambda x: x[df_combined["ACB Score"] != 0].nunique(),
#         ),
#         Earliest_Order_Date=("Order Dt Tm", "min"),
#         Last_Order_Status=("Order Status", "last"),
#     )
#     .reset_index()
# )
summary_df = (
    df_combined.groupby(["Encntr ID", "Visit ID"])
    .agg(
        Unique_Drug_Count=(
            "Order Catalogue Short Description",
            lambda x: x[df_combined["ACB Score"] != 0].nunique(),
        ),
        Earliest_Order_Date=("Order Dt Tm", "min"),
        Last_Order_Status=("Order Status", "last"),
    )
    .reset_index()
)

# %%


def get_acb_score():
    """
    create a dictionary for Total_ACB_Score for each encounter
    from df_combined dataframe

    for each encounter, get the unique values of the Order Catalogue Short Description
    then map the ACB score to each unique value
    and sum that as the Total_ACB_Score

    """
    total_acb_score = {}
    for _, encounter in df_combined.groupby("Encntr ID"):
        unique_drugs = encounter["Order Catalogue Short Description"].unique()
        acb_score = sum(
            drug_df.loc[drug_df["Drug Name"].isin(unique_drugs), "ACB Score"]
        )
        total_acb_score[encounter["Encntr ID"].iloc[0]] = acb_score

    return total_acb_score


# %%
"""
add the Total_ACB_Score to the summary_df
using the dictionary result from get_acb_score()
"""

total_acb_score = get_acb_score()
summary_df["Total_ACB_Score"] = summary_df["Encntr ID"].map(total_acb_score)

# %%
summary_df.to_csv(SAVE_PATH / "Pharmacy_Orders_Processed.csv", index=False)
# %%

# %%
# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
from pathlib import Path

SOURCE_PATH = Path("path/to/raw_data")
SAVE_PATH = Path("./output_dir")
SAVE_PATH.mkdir(exist_ok=True)


# %%
# Cell 2: Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


df_patient = pd.read_excel(SOURCE_PATH / "EMR patient_listing_____from_2018_.xlsx")
df_surgical = pd.read_excel(SOURCE_PATH / "EMR surgical_info_____from_2018.xlsx")

# %%
# Cell 3: Filter data

df_patient = df_patient[
    (df_patient["Hospital"] == "EHS Box Hill")
    & (df_patient["Admission Datetime"] >= "2023-01-01")
    & (df_patient["Admission Datetime"] <= "2024-11-01")
]

df_surgical = df_surgical[
    (df_surgical["Admission Datetime"] >= "2023-01-01")
    & (df_surgical["Admission Datetime"] <= "2024-11-01")
]

# %%
"""
drop duplicates for the surgical dataframe
"""

df_surgical = df_surgical.drop_duplicates()


# %%
"""
    Process patient demographic data by converting data types, filtering records, 
    and calculating new attributes.

    This function performs the following operations:
    1. Converts 'Age' to a numeric data type and encounter-related IDs ('Encntr ID', 
       'Visit No', 'Urn') to integers.
    2. Excludes records with specific discharge dispositions (e.g., Posthumous Organ 
       Procurement, Short Term Transfer, Statistical Separation).
    3. Calculates the Length of Stay (LOS) by computing the difference between 
       'Discharge Datetime' and 'Admission Datetime'.
    4. Classifies the admission type as either 'Surgical' or 'Medical' based on the 
       medical service provided during the encounter. Surgical services are determined 
       by matching keywords such as 'orthopaedics', 'gen surg', and other relevant 
       specialties.

    Parameters:
    -----------
    df_patient : pd.DataFrame
        A DataFrame containing patient demographic and encounter data. The following 
        columns are expected:
        - 'Age'
        - 'Encntr ID'
        - 'Visit No'
        - 'Urn'
        - 'Disch Disposition'
        - 'Admission Datetime'
        - 'Discharge Datetime'
        - 'Medical Service Last'
    
    Returns:
    --------
    pd.DataFrame
        A processed DataFrame with updated data types, filtered records, new 'LOS' 
        (Length of Stay) column, and 'Admission Type' indicating 'Surgical' or 'Medical'.
    
    Notes:
    ------
    - Records with missing or invalid dates are coerced to NaT.
    - Surgical classification is case-insensitive.
    - The function uses regular expressions to match service names.

"""


# Additional Demographic Data Processing
def process_patients(df_patient):
    # Convert to appropriate data types
    df_patient["Age"] = pd.to_numeric(df_patient["Age"], errors="coerce")
    df_patient["Encntr ID"] = df_patient["Encntr ID"].astype(int)
    df_patient["Visit No"] = df_patient["Visit No"].astype(int)
    df_patient["Urn"] = df_patient["Urn"].astype(int)

    # Exclude records with specific discharge disposition
    exclusion_list = [
        "Posthumous Organ Procurement",
        "Short Term Transfer",
        "Statistical Separation",
    ]
    df_patient = df_patient[~df_patient["Disch Disposition"].isin(exclusion_list)]

    # Calculate Length of Stay (LOS)
    df_patient["Admission Datetime"] = pd.to_datetime(df_patient["Admission Datetime"])
    df_patient["Discharge Datetime"] = pd.to_datetime(df_patient["Discharge Datetime"])
    df_patient["LOS"] = (
        df_patient["Discharge Datetime"] - df_patient["Admission Datetime"]
    ).dt.days

    # Exclude admissions with length of stay < 24 hours (LOS in whole days < 1).
    # Episodes with a missing/not-yet-discharged LOS (NaT) are kept here -- they
    # are not excluded for this reason; only known short stays are dropped.
    n_before_los = len(df_patient)
    df_patient = df_patient[df_patient["LOS"].isna() | (df_patient["LOS"] >= 1)]
    print(
        f"  Excluded {n_before_los - len(df_patient)} admissions with LOS < 24 hours "
        f"({len(df_patient)} retained)"
    )

    # Determine Admission Type (Surgical or Medical)
    surgical_services = [
        "orthopaedics",
        "gen surg",
        "vascular",
        "colorectal",
        "uppergi",
        "bariatric",
        "thoracic",
        "breast and endocrine",
        "plastic surgery",
        "opthalmology",
        "urology",
    ]
    df_patient["Admission Type"] = df_patient["Medical Service Last"].apply(
        lambda x: (
            "Surgical"
            if any(re.search(service, str(x).lower()) for service in surgical_services)
            else "Medical"
        )
    )

    return df_patient


df_patient = process_patients(df_patient)

# %%
"""
filtering columns from the patient dataframe:
Admission Datetime
Discharge Datetime
Display
Medical Service Last
Disch Disposition
URN
Age
Sex
Race
Reason for visit
LOS (Length of Stay)
Admission Type 
"""
df_patient_filtered = df_patient[
    [
        "Encntr ID",
        "Visit No",
        "Admission Datetime",
        "Discharge Datetime",
        "Display",
        "Medical Service Last",
        "Disch Disposition",
        "Urn",
        "Age",
        "Sex",
        "Race",
        "Reason For Visit",
        "LOS",
        "Admission Type",
    ]
]

df_patient_filtered = df_patient_filtered.drop_duplicates()

# %%
"""
for df_surgical, we will create a new column with the logic
if Anaesthesia is 'General' then its 'General' otherwise its 'Other'
"""

df_surgical["Anaesthesia Type"] = df_surgical["Anaesthesia"].apply(
    lambda x: "General" if x == "General" else "Other"
)

# %%
df_surgical["Admission Type"] = np.where(
    df_surgical["Anaesthesia Type"].isnull(), "Medical", "Surgical"
)

# %%
df_patient_filtered.to_csv(SAVE_PATH / "EMR_Patient_Processed.csv", index=False)
df_surgical.to_csv(SAVE_PATH / "EMR_Surgical_Processed.csv", index=False)

# ##############################################
# # %%
# """
# left join the two dataframes on Encntr ID,
# left table is patient data and right table is surgical data

# """

# df = pd.merge(df_patient_filtered, df_surgical, on="Encntr ID", how="left")

# # %%
# # if Anaesthesia Type is NaN, then set Admission Type to "Medical",
# # otherweise set to "Surgical"
# # TODO: Check with Swapna:
# # There are some records with NaN in Anaesthesia Type, that already have Admission Type as "Surgical"
# # because of the logic in the above cell
# df["Admission Type"] = np.where(df["Anaesthesia Type"].isnull(), "Medical", "Surgical")

# # %%

# # df.to_csv("verify/EMR_PatientSurgical_Processed.csv", index=False)

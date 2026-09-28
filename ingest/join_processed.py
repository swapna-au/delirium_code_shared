# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
SOURCE_PATH = Path("./output_dir")

# Repo-relative: ingest/join_processed.py -> repo root -> data/. Bronze is
# NOT tracked (contains identifiers) but must land here for 00_prepare_data.py
# / run_all.py to find it, regardless of the CWD this script is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
BRONZE_OUT = REPO_ROOT / "data" / "source_file_bronze.csv"
BRONZE_OUT.parent.mkdir(exist_ok=True)

df_patient = pd.read_csv(SOURCE_PATH / "EMR_Patient_Processed.csv")
df_risks = pd.read_csv(SOURCE_PATH / "EMR_Risks_Processed.csv")
df_vitals = pd.read_csv(SOURCE_PATH / "EMR_Vitals_Processed.csv")
df_creatinine = pd.read_csv(SOURCE_PATH / "EMR_Creatinine_Processed.csv")
df_path_wcc = pd.read_csv(SOURCE_PATH / "EMR_Path_wcc_Processed.csv")
df_path_hb = pd.read_csv(SOURCE_PATH / "EMR_PathHb_Processed.csv")
df_pharmacy_orders = pd.read_csv(SOURCE_PATH / "Pharmacy_Orders_Processed.csv")
df_social_history = pd.read_csv(SOURCE_PATH / "social_history_processed.csv")
df_speech_sensory = pd.read_csv(SOURCE_PATH / "speech_and_sensory_processed.csv")
df_diagnosis = pd.read_csv(SOURCE_PATH / "Diagnosis_Processed.csv")
df_surgical = pd.read_csv(SOURCE_PATH / "EMR_Surgical_Processed.csv")

# %%
df_surgical = df_surgical.drop_duplicates(subset=["Encntr ID", "Visit No"])

# %%
"""
Join multiple DataFrames in a specific order based on the 'N' column.

This script merges several data files (DataFrames) iteratively, ensuring they are joined
in the order specified by the 'N' column. The joining is performed using common columns,
such as 'SITE', 'URN', and 'ENCTR ID', with an outer join to preserve all rows.

Workflow:
1. DataFrames are loaded and assigned an order based on the 'N' column.
2. The files are sorted in ascending order of 'N'.
3. The script iteratively merges
DataFrame Names and Join Order
df_patient_listing (N=1)
df_creatinine (N=2)
df_hb (N=2)
df_wcc (N=2)
df_pharmacy_orders (N=7)
df_emr_risks_details (N=3)
df_social_history_alpha (N=6)
df_speech_and_sensory (N=6)
df_surgical_info (N=6)
df_vitals (N=6)
df_diagnosis (N=5)
Join Logic
The DataFrames must be joined in the order of column N (ascending order).
Use common columns such as SITE, URN, ENCTR ID, VISIT ID, etc., for joining.
Perform an outer join to ensure all data from all DataFrames is retained
"""

df_joined = (
    pd.merge(
        left=df_patient,
        right=df_creatinine,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit ID"],
        how="left",
        suffixes=("_patient", "_creatinine"),
    )
    .merge(
        df_path_hb,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit_ID"],
        how="left",
        suffixes=("_creatinine", "_hb"),
    )
    .merge(
        df_path_wcc,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit_ID"],
        how="left",
        suffixes=("_hb", "_wcc"),
    )
    .merge(
        df_risks,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit ID"],
        how="left",
        suffixes=("_wcc", "_risks"),
    )
    .merge(
        df_vitals,
        left_on=["Encntr ID"],
        right_on=["Encntr ID"],
        how="left",
        suffixes=("_risks", "_vitals"),
    )
    .merge(
        df_diagnosis,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit ID"],
        how="left",
        suffixes=("_vitals", "_diagnosis"),
    )
    .merge(
        df_social_history,
        left_on=["Encntr ID"],
        right_on=["Encntr ID"],
        how="left",
        suffixes=("_diagnosis", "_socials"),
    )
    .merge(
        df_speech_sensory,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit ID"],
        how="left",
        suffixes=("_socials", "_sensory"),
    )
    .merge(
        df_pharmacy_orders,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit ID"],
        how="left",
        suffixes=("_sensory", "_pharmacy"),
    )
    .merge(
        df_surgical,
        left_on=["Encntr ID", "Visit No"],
        right_on=["Encntr ID", "Visit No"],
        how="left",
        suffixes=("_pharmacy", "_surgical"),
    )
)

# %%
# plot missing values of df_joined as heatmap
nan_mask = df_joined.isnull()

fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(nan_mask, cmap="coolwarm", cbar_kws={"ticks": [0, 1]})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


# %%

# Encntr ID, URN, Visit ID columns first
df_joined = df_joined[
    ["Encntr ID", "Urn", "Visit No"]
    + [col for col in df_joined.columns if col not in ["Encntr ID", "URN", "Visit No"]]
]

# df_joined.to_csv(SOURCE_PATH / "EMR_Final_Joined_Processed.csv", index=False)

# %%
# filtered_df = df_joined[
#     (df_joined["Reason For Visit"] != "Cancelled Admit :-")
#     & (df_joined["Display_patient"] not in ["BALFRED BHH", "B1E BHH", "HITH BHH"])
# ]

# "HITH BHH" = Hospital In The Home -- Eastern Health's virtual/outreach care
# model (patients managed remotely rather than as inpatient ward admissions).
# Excluding it is the "virtual care episodes" exclusion described in Methods.
print(f"[COHORT FUNNEL] Base (post-join, pre cancelled/ward filter): {len(df_joined)} rows, "
      f"delirium-positive={int(df_joined['Delirium'].fillna(0).sum())}")
_n_before_ward = len(df_joined)
_n_cancelled = int((df_joined["Reason For Visit"] == "Cancelled Admit :-").sum())
_n_balfred = int((df_joined["Display_patient"] == "BALFRED BHH").sum())
_n_b1e = int((df_joined["Display_patient"] == "B1E BHH").sum())
_n_hith = int((df_joined["Display_patient"] == "HITH BHH").sum())
filtered_df = df_joined[
    (df_joined["Reason For Visit"] != "Cancelled Admit :-")
    & (df_joined["Display_patient"] != "BALFRED BHH")
    & (df_joined["Display_patient"] != "B1E BHH")
    & (df_joined["Display_patient"] != "HITH BHH")  # virtual care (Hospital In The Home)
]
print(f"[COHORT FUNNEL] Excluded: Cancelled Admit={_n_cancelled}, BALFRED BHH={_n_balfred}, "
      f"B1E BHH={_n_b1e}, HITH BHH(virtual care)={_n_hith}  "
      f"(total excluded={_n_before_ward - len(filtered_df)}). Retained: {len(filtered_df)}, "
      f"delirium-positive={int(filtered_df['Delirium'].fillna(0).sum())}")

# %%
# ---------------------------------------------------------------------------
# EXCLUDE ENCOUNTERS WITH A MISSING OUTCOME
# "Delirium" is NaN only when the encounter had ZERO matching rows in the
# diagnosis extract (Diagnosis_Processed.csv) -- i.e. no diagnosis information
# exists for the encounter at all, so the outcome cannot be determined (it is
# NOT known to be "no delirium"; it is unknown). Any case with a missing
# outcome is excluded rather than imputed/assumed negative. This must run
# BEFORE any fillna(0) on Delirium, or the missingness signal is destroyed.
# (The other diagnosis-derived columns -- Dementia, Atrial Fibrillation, COPD,
# Psychiatric disorders, Hip Fracture, Substance Use Disorder -- come from the
# same left join and are NaN for exactly these same encounters.)
# ---------------------------------------------------------------------------
_n_before_missing_outcome = len(filtered_df)
filtered_df = filtered_df[filtered_df["Delirium"].notna()]
print(f"Excluded {_n_before_missing_outcome - len(filtered_df)} admissions with a "
      f"missing delirium outcome (no diagnosis record at all for the encounter). "
      f"{len(filtered_df)} episodes retained.")

# %%
# ---------------------------------------------------------------------------
# DELIRIUM OUTCOME: any diagnosis, any timing.
# An admission-timing (incident vs prevalent) split was investigated and
# reverted -- diagnosis timestamps could not be reliably distinguished from
# discharge-coding lag for the diagnosis types that carry most delirium codes
# (see manuscript Limitations). "Delirium" here reflects any delirium
# diagnosis coded during the encounter. Cases with a missing outcome were
# already excluded above; this cast is now just float -> int (no fillna
# needed since no NaNs remain).
# ---------------------------------------------------------------------------
if "Delirium" in filtered_df.columns:
    filtered_df["Delirium"] = filtered_df["Delirium"].astype(int)

# %%
# ---------------------------------------------------------------------------
# EXCLUDE ACUTE SUBSTANCE USE DISORDER ADMISSIONS
# ingest-diagnosis.py flags encounters with a substance-use-disorder diagnosis
# (alcohol/opioid/sedative/other-drug intoxication, withdrawal, or dependence).
# These are excluded from the cohort per the Methods exclusion criteria.
# Encounters with no diagnosis row have NaN -> fill 0 (not excluded on this basis).
# ---------------------------------------------------------------------------
if "Substance Use Disorder" in filtered_df.columns:
    filtered_df["Substance Use Disorder"] = (
        filtered_df["Substance Use Disorder"].fillna(0).astype(int)
    )
    _n_before_sud = len(filtered_df)
    _n_delirium_before_sud = int(filtered_df["Delirium"].sum())
    _n_sud_and_delirium = int(((filtered_df["Substance Use Disorder"] == 1) & (filtered_df["Delirium"] == 1)).sum())
    filtered_df = filtered_df[filtered_df["Substance Use Disorder"] == 0]
    print(
        f"Excluded {_n_before_sud - len(filtered_df)} admissions with an acute "
        f"substance use disorder diagnosis (of which {_n_sud_and_delirium} were also "
        f"delirium-positive). {len(filtered_df)} episodes retained, "
        f"delirium-positive={int(filtered_df['Delirium'].sum())} "
        f"(was {_n_delirium_before_sud} before this exclusion)."
    )
    filtered_df = filtered_df.drop(columns=["Substance Use Disorder"])


# %%
columns_to_drop = [
    "Visit ID_wcc",
    "Hospital_creatinine",
    "Ward_creatinine",
    "Clinical Unit",
    "Admission Datetime_creatinine",
    "Discharge Datetime_creatinine",
    "Ur_creatinine",
    "Age_creatinine",
    "Sex_creatinine",
    "Clinical Event ID_wcc",
    "Event End Dt Tm Tz",
    "Event Cd",
    "Test",
    "Result Val",
    "Numeric Result",
    "Display_creatinine",
    "Accession Nbr",
    "Order ID",
    "Normalcy",
    "Event Class",
    "Length of Stay",
    "Visit_ID_hb",
    "Encntr_Type",
    "Encntr_Status",
    "Sex_hb",
    "Hospital_hb",
    "Ward_hb",
    "Clinical_Unit_hb",
    "Admission_Datetime",
    "Discharge_Datetime",
    "Ur_hb",
    "Age",
    "Sex_wcc",
    "Visit_ID_wcc",
    "Ward",
    "Clinical_Unit_wcc",
    "UR",
    "Unit",
    "Visit ID_risks",
    "Clinical Event ID_risks",
    "Clinical Event Performed Date Time",
    "Clinical Event Code",
    "BOC Risk",
    "SHX Lives with_question",
    "SHx Carer Status_question",
    "SHx Home Accommodation Ownership_question",
    "Visit ID_sensory",
    "SD Speech Deficit",
    "Visit ID",
    "Earliest_Order_Date",
    "Last_Order_Status",
    "Admission Datetime",
    "Discharge Datetime",
    "Surg Case Nbr Formatted",
    "Priority",
    "Procedure Case Level",
    "Case Completed",
    "Procedure",
    "Procedure Text",
    "Anaesthesia",
    "Anaesthesia Type",
    "Admission Type_surgical",
    "Unit_creatinine",
    "Unit_hb",
    "Visit ID_socials",
    "Ur",
]

# %%
df = filtered_df.drop(columns=columns_to_drop)

# %%
df.to_csv(BRONZE_OUT, index=False)
print(f"[OK] Saved: {BRONZE_OUT}")
# %%

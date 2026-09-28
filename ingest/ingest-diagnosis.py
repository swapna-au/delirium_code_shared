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
DiagDescDict = {
    "Atrial Fibrillation": [
        "Persistent atrial fibrillation",
        "Chronic atrial fibrillation",
        "Unspecified atrial fibrillation",
        "Permanent atrial fibrillation",
        "Atrial fibrillation",
        "Paroxysmal atrial fibrillation",
        "Atrial fibrillation with rapid ventricular response",
        "Atrial fibrillation and atrial flutter, unspecified",
        "Paroxysmal AF (atrial fibrillation) with RVR (rapid ventricular response)",
        "Paroxysmal a-fib (atrial fibrillation) with RVR (rapid ventricular response)",
        "AF - Atrial fibrillation",
        "Rapid atrial fibrillation",
        "Controlled atrial fibrillation",
        "AF - Paroxysmal atrial fibrillation",
        "PAF - Paroxysmal atrial fibrillation",
        "Atrial fibrillation and flutter",
        "Atrial fibrillation or atrial flutter",
        "Atrial fibrillation due to heart valve disorder",
    ],
    "COPD": [
        "Other chronic obstructive pulmonary disease",
        "Chronic obstructive pulmonary disease with acute lower respiratory infection",
        "Chronic obstructive pulmonary disease with (acute) exacerbation",
        "Chronic obstructive pulmonary disease, unspecified",
        "Emphysema, unspecified",
        "Unilateral pulmonary emphysema [MacLeod’s syndrome]",
        "Chronic obstructive lung disease",
        "Chronic obstructive pulmonary disease with acute exacerbation",
        "Chronic obstructive pulmonary disease with lower respiratory infection",
        "Emphysema",
        "Chronic obstructive bronchitis",
        "Chronic obstructive pulmonary disease with acute exacerbation, unspecified",
        "Other specified chronic obstructive pulmonary disease",
        "Emphysema, without mention of chronic obstructive pulmonary disease",
        "Chronic obstructive pulmonary disease",
        "Asthma, without mention of chronic obstructive pulmonary disease",
        "Asthma-chronic obstructive pulmonary disease overlap syndrome",
        "COPD (chronic obstructive pulmonary disease) with acute lower respiratory infection",
        "COPD - Chronic obstructive pulmonary disease",
        "Acute exacerbation of COPD (chronic obstructive pulmonary disease) with asthma",
        "Acute exacerbation of chronic obstructive pulmonary disease",
        "Chronic obstructive pulmonary disease (COPD), with acute lower respiratory infection",
        "Chronic obstructive pulmonary disease (COPD), other",
        "Chronic obstructive pulmonary disease (COPD), with acute exacerbation",
        "Acute exacerbation of COPD (chronic obstructive pulmonary disease)",
        "Chronic Obstructive Pulmonary Disease, Major Complexity",
        "Chronic Obstructive Pulmonary Disease, Minor Complexity",
        "Interstitial emphysema",
        "Pulmonary emphysema",
        "Emphysema of lung",
        "Interstitial emphysema of lung",
        "Emphysematous bulla",
        "Bullous emphysema",
        "CPFE - combined pulmonary fibrosis and emphysema",
        "Unspecified chronic bronchitis",
        "Chronic bronchitis",
        "Asthma-COPD overlap syndrome (ACOS)",
        "COPD co-occurrent with acute bronchitis",
        "At risk of COPD",
        "Severe COPD",
        "COPD with acute lower respiratory infection",
        "Acute infective exacerbation of COPD",
        "COPD",
        "Acute exacerbation of COPD",
        "CHRNIC OBSTRCT AIRWAY DIS +CCC",
    ],
    "Psychiatric disorders": [
        "Manic episode",
        "Bipolar disorder",
        "Major depressive disorder, single episode",
        "Major depressive disorder, recurrent",
        "Persistent mood disorders, such as dysthymia and cyclothymia",
        "Phobic anxiety disorders (e.g., social phobia, agoraphobia)",
        "Other anxiety disorders (e.g., generalized anxiety disorder, panic disorder)",
        "Obsessive-compulsive disorder (OCD)",
        "Schizophrenia",
        "Persistent delusional disorders",
        "Schizoaffective disorders",
        "Specific personality disorders (e.g., borderline, antisocial)",
        "Mixed and other personality disorders",
        "Enduring personality changes, not attributable to brain damage and disease",
        "Major depressive disorder",
        "Dysthymia",
        "Generalized anxiety disorder",
        "Obsessive-compulsive disorder",
        "Panic disorder",
        "Schizoaffective disorder",
        "Paranoid personality disorder",
        "Borderline personality disorder",
        "Antisocial personality disorder",
        "Avoidant personality disorder",
        "Autism spectrum disorder",
        "Attention-deficit hyperactivity disorder",
        "Anxiety",
        "Panic disorder [episodic paroxysmal anxiety]",
        "Generalised anxiety disorder",
        "Mixed anxiety and depressive disorder",
        "Other specified anxiety disorders",
        "Anxiety disorder, unspecified",
        "Anxiety Disorders, Minor Complexity",
        "Anxiety Disorders, Major Complexity",
        "Adjustment disorder with anxiety",
        "Adjustment disorder with mixed anxiety and depressed mood",
        "Social anxiety disorder",
        "GAD - Generalised anxiety disorder",
        "Anxiety state",
        "Anxiety depression",
        "Anxiety disorder",
        "Anxiety neurosis",
        "Anxiety in dementia",
        "Severe major depressive disorder co-occurrent with anxiety single episode",
        "Anxiety attack",
        "Anxiety disorder, generalised or phobic",
        "Anxiety disorder with mixed depressive disorder",
        "Schizoaffective disorder, manic type",
        "Manic episode, unspecified",
        "Bipolar affective disorder, current episode hypomanic",
        "Bipolar affective disorder, current episode manic without psychotic symptoms",
        "Bipolar affective disorder, current episode manic with psychotic symptoms",
        "Bipolar 1 disorder, single manic episode",
        "Bipolar disorder, most recent episode manic",
        "Manic bipolar I disorder",
        "Manic psychosis",
        "Single manic episode, severe, with psychosis",
        "Bipolar affective disorder, current episode manic",
        "Bipolar affective disorder, currently manic, severe, with psychosis",
        "Bipolar affective disorder, currently manic, in full remission",
        "Bipolar I disorder, most recent episode manic",
        "Manic symptoms with primary psychotic disorder",
        "Manic disorder, single episode",
        "Hypomanic behaviour",
        "Manic disorder",
        "Manic episode, single",
        "Depressive episode, unspecified, not specified as arising in the postnatal period",
        "Depression",
        "Depressed mood",
        "Depressive episode",
        "Depressive symptoms",
        "Depressed bipolar I disorder",
        "Depressive disorder",
        "Depressive neurosis",
        "Depressive disorder, recurrent (except in bipolar or postnatal depression)",
        "Depressive episode, single (except postnatal depression)",
        "Other obsessive-compulsive disorders",
        "Obsessive-compulsive disorder, unspecified",
        "Eating and Obsessive-Compulsive Disorders, Major Complexity",
        "Eating and Obsessive-Compulsive Disorders, Minor Complexity",
        "OCD - Obsessive-compulsive disorder",
        "Organic personality disorder",
        "Dissocial personality disorder",
        "Emotionally unstable personality disorder, borderline type",
        "Anankastic personality disorder",
        "Anxious [avoidant] personality disorder",
        "Dependent personality disorder",
        "Other specific personality disorders",
        "Personality disorder, unspecified",
        "Personality Disorders and Acute Reactions, Minor Complexity",
        "Personality Disorders and Acute Reactions, Major Complexity",
        "Schizotypal personality disorder",
        "Cluster C personality disorder",
        "Cluster B personality disorder",
        "Cluster A personality disorder",
        "Narcissistic personality disorder",
        "Borderline personality disorder in remission",
        "BPD - borderline personality disorder",
        "Schizophrenia Disorders W/O Mental Health Legal Status",
        "Paranoid schizophrenia",
        "Catatonic schizophrenia",
        "Other schizophrenia",
        "Schizophrenia, unspecified",
        "Schizotypal disorder",
        "Acute schizophrenia-like psychotic disorder with mention of associated acute stress",
        "Schizoaffective disorder, depressive type",
        "Schizoaffective disorder, mixed type",
        "Other schizoaffective disorders",
        "Schizoaffective disorder, unspecified",
        "Late onset schizophrenia",
        "Schizoaffective disorder, bipolar type",
        "Chronic residual schizophrenia with acute exacerbations",
        "Chronic schizophrenia",
        "Chronic paranoid schizophrenia",
        "Subchronic paranoid schizophrenia",
        "Acute exacerbation of chronic schizophrenia",
        "Schizoaffective schizophrenia",
        "Acute exacerbation of chronic schizoaffective schizophrenia",
        "Schizoaffective disorder, with or without psychotic episode",
        "Persistent delusional disorder and paranoia (except in schizophrenia)",
        "Schizophrenia Disorders, Major Complexity",
        "Schizophrenia Disorders, Minor Complexity",
        "Schizophrenia Disorders W Mental Health Legal Status",
        "Agoraphobia without mention of panic disorder",
        "Agoraphobia with panic disorder",
        "Panic attack",
        "Paranoid delusion",
        "Paranoia",
        "Paranoid psychosis",
        "Paranoia and Acute Psychotic Disorders, Major Complexity",
        "Paranoia and Acute Psychotic Disorders, Minor Complexity",
        "Psychosis",
    ],
    "Hip Fracture": [
        "Fracture of femur",
        "Fracture of neck of femur",
        "Pertrochanteric fracture",
        "Subtrochanteric fracture",
        "Fracture of shaft of femur",
        "Fracture of lower end of femur",
        "Multiple fractures of femur",
        "Other fractures of femur",
        "Unspecified fracture of femur",
        "shaft of femur",
        "NOF",
        "Distal end of femur",
        "Proximal end of femur",
        "FRACTURE NECK FEMUR+CSCC",
        "FRACTURE OF NECK FEMUR-CSCC",
        "Fracture of other parts of neck of femur",
        "Fracture of trochanteric section of femur, unspecified",
        "Fracture of intertrochanteric section of femur",
        "Fracture of shaft of femur",
        "Fracture of lower end of femur, part unspecified",
        "Fracture of lower epiphysis (separation) of femur",
        "Supracondylar fracture of femur",
        "Intercondylar fracture of femur",
        "Fracture of femur, part unspecified",
        "Fracture of neck of femur, part unspecified",
        "Fracture of intracapsular section of femur",
        "Fracture of subcapital section of femur",
        "Fracture of midcervical section of femur",
        "Fracture of base of neck of femur",
        "Sequelae of fracture of femur",
        "Pathological fracture of femur",
        "Fracture of trochanter of femur",
        "Sarcoma of femur",
        "Closed subcapital fracture of femur",
        "Closed fracture of greater trochanter of femur",
        "Periprosthetic fracture of proximal femur",
        "Open fracture of head of right femur",
        "Closed fracture of trochanter of right femur",
        "Open fracture of distal end of right femur",
        "Closed subcapital fracture of neck of femur",
        "Closed fracture of head of right femur",
        "Closed fracture of base of neck of femur",
        "Closed fracture of femur",
        "Fracture of neck of femur",
        "Trochanteric fracture of neck of femur",
        "Intertrochanteric fracture of femur",
        "Closed fracture of head of left femur",
        "Open fracture of base of neck of left femur",
        "Atypical fracture of femur",
        "Stress fracture of femur",
        "Closed fracture of trochanter of left femur",
        "Fracture of proximal end of femur",
        "Intracapsular fracture of neck of femur",
        "Subcapital fracture of neck of femur",
        "Transcervical fracture of neck of femur",
        "Basicervical fracture of neck of femur",
        "Subtrochanteric fracture of femur",
        "Open fracture of neck of femur",
        "Closed fracture of neck of femur",
        "Closed fracture of femur, subcapital",
        "NOF - Fracture of neck of femur",
        "Fracture of femur, other or multiple parts",
        "Fracture of femur, neck or head (proximal)",
        "Open fracture of base of neck of right femur",
        "Pertrochanteric fracture of neck of femur",
        "Fractures of Neck of Femur, Major Complexity",
        "Fractures of Neck of Femur, Minor Complexity",
        "Hip fracture",
        "Fracture of hip",
        "Fracture of bone of hip region",
        "OTHER HIP & FEMUR PR -CCC",
    ],
    "Delirium": [
        "Delirium, not induced by alcohol and other psychoactive substances",
        "Delirium not superimposed on dementia, so described",
        "Delirium superimposed on dementia",
        "Other delirium",
        "Delirium, unspecified",
        "Disorientation, unspecified",
        "Encephalopathy, unspecified",
        "Delirium",
        "Acute delirium",
        "Postoperative delirium",
        "Delirium tremens",
        "Delirium due to drugs",
        "Alzheimer's disease co-occurrent with delirium",
        "Delirium management",
        "Delirium due to multiple aetiological factors",
        "Hypoactive postoperative delirium",
        "Dementia of the Alzheimer's type, with late onset, with delirium",
        "Vascular dementia, with delirium",
        "Post-injection delirium sedation syndrome",
        "Delirium of mixed origin",
        "Delirium co-occurrent with dementia",
        "Dementia Alzheimers type, late onset with delirium",
        "Delirium following surgical procedure",
        "Hypoactive delirium following surgical procedure",
        "Hyperactive postoperative delirium",
        "Delirium caused by drug",
        "Delirium, organic, not dementia or alcohol or drugs",
        "Delirium, acute",
        "Delirium due to disease or infection",
        "Delirium due to dementia",
        "Delirium, Major Complexity",
        "Delirium, Minor Complexity",
        "Disorientation",
        "Disorientation or confusion",
        "DELIRIUM-CCC",
        "DELIRIUM+CCC",
    ],
    "Dementia": [
        "dementia and other chronic disturbances of cerebral function, minor complexity",
        "dementia and other chronic disturbances of cerebral function, major complexity",
        "dementia in parkinsons disease",
        "vascular dementia",
        "dementia with behavioural disturbance",
        "altered behaviour in huntington's dementia",
        "alzheimer dementia",
        "bpsd - behavioural and psychological symptoms of dementia",
        "agitation due to dementia",
        "mvad - mixed vascular alzheimer dementia",
        "aggression due to dementia",
        "dementia with mixed aetiology",
        "dementia associated with parkinson's disease",
        "dementia of the alzheimer's type, with late onset, with delirium",
        "vascular dementia, with delirium",
        "dementia",
        "alcohol-induced persisting dementia",
        "dementia of the alzheimers type with early onset",
        "mixed dementia",
        "frontotemporal dementia",
        "delirium co-occurrent with dementia",
        "behavioural variant of frontotemporal dementia",
        "behavioural and psychological symptoms of dementia",
        "anxiety in dementia",
        "lewy body dementia with behavioural disturbance",
        "aggression in dementia",
        "dementia in alzheimer's disease with late onset",
        "alzheimers dementia, late onset, with delusions",
        "dementia alzheimers type, late onset with delirium",
        "early onset dementia",
        "dementia of the lewy body type",
        "alcoholic dementia",
        "rapidly progressive dementia",
        "agitation in dementia",
        "dementia of frontal lobe type",
        "vascular dementia with behavioural disturbance",
        "behavioural disturbance due to multi-infarct dementia",
        "dementia in alzheimer's disease, atypical or mixed type, with psychological or behavioural disturbance (g30.8+)",
        "dementia in alzheimer's disease with early onset, without mention of psychological or behavioural disturbance (g30.0+)",
        "dementia in alzheimer's disease, unspecified, with psychological or behavioural disturbance (g30.9+)",
        "unspecified dementia, with psychological or behavioural disturbance",
        "vascular dementia, unspecified, without mention of psychological or behavioural disturbance",
        "dementia, unspecified",
        "dementia in alzheimers disease with early onset, with psychological or behavioural disturbance (g30.0+)",
        "dementia in alzheimers disease, atypical or mixed type, with psychological or behavioural disturbance (g30.8+)",
        "dementia in alzheimers disease, unspecified, without mention of psychological or behavioural disturbance (g30.9+)",
        "dementia in parkinsons disease, with psychological or behavioural disturbance (g20+)",
        "dementia in parkinsons disease, without mention of psychological or behavioural disturbance (g20+)",
        "unspecified dementia, without mention of psychological or behavioural disturbance",
        "dementia in other specified diseases classified elsewhere, with psychological or behavioural disturbance",
        "other vascular dementia, with psychological or behavioural disturbance",
        "mixed cortical and subcortical vascular dementia, with psychological or behavioural disturbance",
        "vascular dementia, unspecified, with psychological or behavioural disturbance",
        "dementia in picks disease, with psychological or behavioural disturbance (g31.0+)",
        "dementia in other specified diseases classified elsewhere, without mention of psychological or behavioural disturbance",
        "dementia in alzheimers disease with early onset, without mention of psychological or behavioural disturbance (g30.0+)",
        "dementia in picks disease, without mention of psychological or behavioural disturbance (g31.0+)",
        "dementia in alzheimers disease, atypical or mixed type, without mention of psychological or behavioural disturbance (g30.8+)",
        "dementia in alzheimers disease, unspecified, with psychological or behavioural disturbance (g30.9+)",
        "multi-infarct dementia, with psychological or behavioural disturbance",
        "other vascular dementia, without mention of psychological or behavioural disturbance",
        "dementia in huntingtons disease, without mention of psychological or behavioural disturbance (g10+)",
        "dementia without mention of psychological or behavioural disturbance (except in alzheimer's disease)",
        "dementia with psychological or behavioural disturbance (except in alzheimer's disease)",
        "delirium due to dementia",
        "dementia in alzheimer's disease, unspecified, without mention of psychological or behavioural disturbance (g30.9+)",
        "dementia in pick's disease, with psychological or behavioural disturbance (g31.0+)",
        "dementia in alzheimer's disease with early onset, with psychological or behavioural disturbance (g30.0+)",
        "dementia in parkinson's disease, with psychological or behavioural disturbance (g20+)",
    ],
    # Used as a COHORT EXCLUSION criterion (acute substance use disorder
    # admissions), not as a comorbidity predictor. See join_processed.py.
    "Substance Use Disorder": [
        "Mental and behavioural disorders due to use of alcohol",
        "Mental and behavioural disorders due to use of alcohol, acute intoxication",
        "Mental and behavioural disorders due to use of alcohol, withdrawal state",
        "Mental and behavioural disorders due to use of alcohol, withdrawal state with delirium",
        "Mental and behavioural disorders due to use of alcohol, dependence syndrome",
        "Mental and behavioural disorders due to use of alcohol, harmful use",
        "Alcohol dependence",
        "Alcohol dependence, uncomplicated",
        "Alcohol withdrawal",
        "Alcohol withdrawal delirium",
        "Alcohol intoxication",
        "Alcohol intoxication, uncomplicated",
        "Acute alcoholic intoxication",
        "Alcohol abuse",
        "Alcohol use disorder",
        "Harmful use of alcohol",
        "Toxic effect of alcohol",
        "Mental and behavioural disorders due to use of opioids",
        "Mental and behavioural disorders due to use of opioids, acute intoxication",
        "Mental and behavioural disorders due to use of opioids, withdrawal state",
        "Mental and behavioural disorders due to use of opioids, dependence syndrome",
        "Mental and behavioural disorders due to use of opioids, harmful use",
        "Opioid dependence",
        "Opioid withdrawal",
        "Opioid intoxication",
        "Opioid abuse",
        "Opioid use disorder",
        "Mental and behavioural disorders due to use of sedatives or hypnotics",
        "Mental and behavioural disorders due to use of sedatives or hypnotics, withdrawal state",
        "Mental and behavioural disorders due to use of sedatives or hypnotics, dependence syndrome",
        "Sedative, hypnotic or anxiolytic-related disorder",
        "Benzodiazepine dependence",
        "Benzodiazepine withdrawal",
        "Mental and behavioural disorders due to multiple drug use and use of other psychoactive substances",
        "Mental and behavioural disorders due to use of cannabinoids",
        "Mental and behavioural disorders due to use of cocaine",
        "Mental and behavioural disorders due to use of other stimulants, including caffeine",
        "Mental and behavioural disorders due to use of amphetamines",
        "Amphetamine dependence",
        "Amphetamine intoxication",
        "Amphetamine withdrawal",
        "Stimulant use disorder",
        "Mental and behavioural disorders due to use of hallucinogens",
        "Mental and behavioural disorders due to use of volatile solvents",
        "Substance-induced delirium",
        "Substance use disorder",
        "Drug dependence",
        "Drug withdrawal",
        "Drug abuse",
        "Polysubstance abuse",
        "Polysubstance dependence",
        "SUD - Substance use disorder",
    ],
}

# %%
# replace all entries in the dictionary to lowercase
for key, value in DiagDescDict.items():
    DiagDescDict[key] = [x.lower() for x in value]


# %%
# Cell 2: Load the data
df = pd.read_excel(SOURCE_PATH / "EMR_diagnosis_final_extract_from_2021.xlsx")

# %%
"""
drop the Unnamed: 0 column
"""
# df.drop("Unnamed: 0", axis=1, inplace=True)

# %%
"""
Rename Visit No to Visit ID
Drop duplicate Diagnosis ID
"""

df.rename(columns={"Visit No": "Visit ID"}, inplace=True)
df.drop_duplicates(subset=["Diagnosis ID"], inplace=True)

# %%
# NOTE: the cohort date-range filter (2023-01-01 to 2024-11-01) is applied
# further below, after Admission Datetime is merged in from the patient
# listing extract (this raw diagnosis file has no Admission Datetime or
# Hospital column of its own -- see the "Date-range filter" block near the
# incident-delirium timing logic).


# %%
"""
Make the following new columns where default value is 0
'Atrial Fibrillation'
'DELIRIUM'
'COPD'	
'Psychiatric disorders'
'HIP FRACTURE'
"""
df["Atrial Fibrillation"] = 0
df["Delirium"] = 0
df["COPD"] = 0
df["Psychiatric disorders"] = 0
df["Hip Fracture"] = 0
df["Dementia"] = 0
df["Substance Use Disorder"] = 0


# %%
"""
Based on the dictionary above, write a function
that will take the Diagnosis Description and the key as parameters, 
and return 1 if the description is in the dictionary
"""


def check_diag_desc(diag_desc, key):
    if diag_desc.lower() in DiagDescDict[key]:
        return 1
    else:
        return 0


# %%
"""
for all the columns in the dictionary above,
update the columns using an apply and lambda function
while passing the key to the function

df["Atrial Fibrillation"] 
df["Delirium"] 
df["COPD"] 
df["Psychiatric disorders"]
df["Hip Fracture"]
df["Dementia"]
"""

df["Atrial Fibrillation"] = df["Diagnosis Description"].apply(
    lambda x: check_diag_desc(x, "Atrial Fibrillation")
)

df["Delirium"] = df["Diagnosis Description"].apply(
    lambda x: check_diag_desc(x, "Delirium")
)

df["COPD"] = df["Diagnosis Description"].apply(lambda x: check_diag_desc(x, "COPD"))

df["Psychiatric disorders"] = df["Diagnosis Description"].apply(
    lambda x: check_diag_desc(x, "Psychiatric disorders")
)

df["Hip Fracture"] = df["Diagnosis Description"].apply(
    lambda x: check_diag_desc(x, "Hip Fracture")
)

df["Dementia"] = df["Diagnosis Description"].apply(
    lambda x: check_diag_desc(x, "Dementia")
)

df["Substance Use Disorder"] = df["Diagnosis Description"].apply(
    lambda x: check_diag_desc(x, "Substance Use Disorder")
)


# %%
# ---------------------------------------------------------------------------
# INCIDENT-DELIRIUM TIMING
# The delirium OUTCOME must capture only NEW-ONSET (incident) delirium that
# began AFTER the first 48 hours of admission. This was investigated and
# reverted: Diagnosis Dt Tm timestamps for the diagnosis types that carry most
# delirium codes (Principal Dx / Additional Dx) are frequently close to the
# discharge date (58.7% within 24h of discharge, median 9.7h before discharge)
# and cannot be reliably distinguished from documentation/coding lag vs true
# clinical onset. Restricting to diagnosis types with more reliable real-time
# timestamps (Clinical Dx / Complication) leaves too few cases to be usable
# (68 rows total). Given this, the timing split is not reliably identifiable
# from this data, so ALL delirium diagnoses (any timing) are retained as
# outcome-positive, matching the original (pre-48h-window) definition. See the
# manuscript Limitations section for the full analysis supporting this call.
#
# Requires:
#   - "Diagnosis Dt Tm"    -> onset date of each diagnosis (this extract)
#   - "Admission Datetime" -> merged in from the patient listing extract
# ---------------------------------------------------------------------------

df["Diagnosis Dt Tm"] = pd.to_datetime(df["Diagnosis Dt Tm"], errors="coerce")

df_adm = pd.read_excel(
    SOURCE_PATH / "EMR patient_listing_____from_2018_.xlsx",
    usecols=["Encntr ID", "Admission Datetime"],
)
df_adm["Admission Datetime"] = pd.to_datetime(df_adm["Admission Datetime"], errors="coerce")
df_adm = df_adm.drop_duplicates(subset=["Encntr ID"])
df = df.merge(df_adm, on="Encntr ID", how="left")

# Cohort date-range filter (same window as every other ingest source). The raw
# diagnosis extract spans back to 2021, well before the study window.
_n_before_daterange = len(df)
df = df[
    (df["Admission Datetime"] >= "2023-01-01")
    & (df["Admission Datetime"] <= "2024-11-01")
]
print(f"Date-range filter: {_n_before_daterange - len(df)} diagnosis rows excluded "
      f"(admission outside 2023-01-01 to 2024-11-01); {len(df)} retained.")

# %%
# Group by 'Encntr ID' and apply logical OR operation for specified columns.
# All flags (comorbidities + Delirium) are timing-agnostic (any diagnosis
# coded at any point in the encounter). Substance Use Disorder is a COHORT
# EXCLUSION criterion, not a predictor -- carried through the same way and
# dropped in join_processed.py.
columns_to_or = [
    "Atrial Fibrillation",
    "COPD",
    "Psychiatric disorders",
    "Hip Fracture",
    "Dementia",
    "Delirium",
    "Substance Use Disorder",
]


def logical_or(group):
    return group.max()


# %%
summary_df = (
    df.groupby(["Encntr ID", "Visit ID"])[columns_to_or]
    .apply(logical_or)
    .reset_index()
)

print(
    f"Delirium (outcome=1, any timing): {int(summary_df['Delirium'].sum())} episodes | "
    f"Substance use disorder flagged (excluded downstream): {int(summary_df['Substance Use Disorder'].sum())}"
)

# %%
summary_df.to_csv(SAVE_PATH / "Diagnosis_Processed.csv", index=False)
# %%

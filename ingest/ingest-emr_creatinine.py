# %%
# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: EDA plots below are diagnostic only, not saved to disk
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from pathlib import Path

SOURCE_PATH = Path("path/to/raw_data")
SAVE_PATH = Path("./output_dir")
SAVE_PATH.mkdir(exist_ok=True)


# %%
# Cell 2: Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Load the data
df = pd.read_excel(SOURCE_PATH / "EMR Path creatinine_____from_jan_2023.xlsx")


# %%
# Cell 3: Display the first few rows of the data
def display_data(data):
    print(data.head())


# %%
# Cell 4: Handle missing values
# def handle_missing_values(data):
#     data = data.fillna(method="ffill").fillna(method="bfill")
#     return data


# Filter data
df = df[
    (df["Hospital"] == "EHS Box Hill")
    & (df["Admission Datetime"] >= "2023-01-01")
    & (df["Admission Datetime"] <= "2024-11-01")
]


# %%
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


df["Result Val"] = df.apply(convert_result_val, axis=1)

# %%
# Review and handle outliers (don't remove - just flag them)
# TODO: move to a separate file for validation
# Assuming outliers are handled by removing them if they are beyond 3 standard deviations
df = df[
    (df["Result Val"] >= df["Result Val"].mean() - 3 * df["Result Val"].std())
    & (df["Result Val"] <= df["Result Val"].mean() + 3 * df["Result Val"].std())
]

# %%
# Group by Encounter ID and calculate mean result value
df_mean = df.groupby("Encntr ID").agg({"Result Val": "mean"}).reset_index()
df_mean.rename(columns={"Result Val": "Mean Cr Result Value"}, inplace=True)

# %%
# Merge with original data to keep other columns
df = df.drop_duplicates(subset=["Encntr ID"])
df = df.merge(df_mean, on="Encntr ID")

# %%

# Convert Result Val to mg/dl
df["Cr Result Val mg/dl"] = df["Mean Cr Result Value"] / 88.42
# Denote positive or negative Creatinine value
df["Cr Positive"] = df["Cr Result Val mg/dl"].apply(
    lambda x: "Positive" if x > 1.5 else "Negative"
)


# %%
# Display basic info and check for nulls
print(df.info())
print(df.describe())
print(df.isnull().sum())


# %%
# Cell 6: Normalize/Standardize data (example: scaling numerical columns)
def normalize_data(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[
        numerical_cols
    ].std()
    return data


# %%
# Calculate length of stay (rename with the source file reference)
df["Length of Stay"] = (df["Discharge Datetime"] - df["Admission Datetime"]).dt.days


# %%
# Plot distribution of length of stay
plt.figure(figsize=(10, 6))
sns.histplot(df["Length of Stay"], kde=True, bins=30)
plt.title("Distribution of Length of Stay")
plt.xlabel("Length of Stay (days)")
plt.ylabel("Frequency")
plt.show()

# %%
# cange to boxplot
# Check Cr Positive distribution
plt.figure(figsize=(8, 5))
sns.histplot(x="Cr Positive", data=df)
plt.title("Cr Positive Count")
plt.show()

# %%
# Analyze Ward and Unit distributions
plt.figure(figsize=(12, 6))
sns.countplot(y="Ward", data=df, order=df["Ward"].value_counts().index)
plt.title("Admissions by Ward")
plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.countplot(
    y="Clinical Unit", data=df, order=df["Clinical Unit"].value_counts().index
)
plt.title("Admissions by Clinical Unit")
plt.show()
# %%
# Box plot for Cr Result Value by Ward
plt.figure(figsize=(14, 7))
sns.boxplot(x="Ward", y="Cr Result Val mg/dl", data=df)
plt.xticks(rotation=45)
plt.title("Creatinine Result by Ward")
plt.show()
# %%
# Correlation heatmap for numeric fields
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=["float64", "int64"]).corr()
sns.heatmap(numeric_cols, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# %%
# Scatter plot for Mean Cr Result vs Length of Stay
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Mean Cr Result Value", y="Length of Stay", hue="Cr Positive", data=df
)
plt.title("Mean Creatinine Result vs Length of Stay")
plt.show()

# %%
df.to_csv(SAVE_PATH / "EMR_Creatinine_Processed.csv", index=False)
# %%

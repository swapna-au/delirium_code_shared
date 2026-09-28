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
df = pd.read_csv(SOURCE_PATH / "social_history_alpha______from_jan_2018.csv")

# %%
"""
check the data for unique comibation of Encntr ID and UR
"""
df_check = df.groupby(["Encntr ID", "Ur"]).size().reset_index()

# %%
"""
only interested in records with the following Question:
'SHx Home Accommodation Ownership'
'SHX Lives with'
'SHx Carer Status'


"""

df_social_history = df[
    df["Question"].isin(
        [
            "SHx Home Accommodation Ownership",
            "SHX Lives with",
            "SHx Carer Status",
        ]  # filter for the required questions
    )
]

# # %%
# """
# for Question == 'SHX Lives with',
# transform the Answer information to:
# 0 - if the patient lives alone
# 1 - if the patient lives with someone
# nan - if blank
# """

# lives_with_someone = {
#     "Carer",
#     "Children",
#     "Father",
#     "Mother",
#     "Siblings",
#     "Significant other",
#     "Spouse",
# }


# def update_lives_with(row):
#     if row["Question"] == "SHX Lives with":
#         if type(row["Answer"]) == str and row["Answer"].strip() == "Alone":
#             return 0
#         elif type(row["Answer"]) == str and row["Answer"].strip() in lives_with_someone:
#             return 1
#         else:
#             return np.nan
#     return row["Answer"]


# # %%
# df_social_history["Answer2"] = df_social_history.apply(update_lives_with, axis=1)

# %%
"""
update the answer column, if it of time str, strip for white spaces
"""
df_social_history["Answer"] = df_social_history["Answer"].apply(
    lambda x: x.strip() if type(x) == str else x
)

# %%
"""
for each of the questions, there is an order of priority
blanks should be coded as -1
if there are multiple records for the same encounter ID and UR,
then take the maximum value using the following logic:
For SHx Home Accommodation Ownership - (priority lowest to highest)
Disability Accommodation
Emergency Accomodation   
Housing Victoria
MOH                      
Private Rental           
Privately Owned Home     
Residential Aged Care    
Retirement Village       

For SHX Lives with - (priority lowest to highest)
Alone                    
Carer                    
Children                 
Father                   
Mother                   
Siblings                 
Significant other        
Spouse                   

For SHx Carer Status - (priority lowest to highest)
Carer for others         
Full time carer          
No carer                 
Paid carer               
Unpaid/informal carer    

"""


def update_answer(row):
    if row["Question"] == "SHx Home Accommodation Ownership":
        if row["Answer"] == "Disability Accommodation":
            return 0
        elif row["Answer"] == "Emergency Accomodation":
            return 1
        elif row["Answer"] == "Housing Victoria":
            return 2
        elif row["Answer"] == "MOH":
            return 3
        elif row["Answer"] == "Private Rental":
            return 4
        elif row["Answer"] == "Privately Owned Home":
            return 5
        elif row["Answer"] == "Residential Aged Care":
            return 6
        elif row["Answer"] == "Retirement Village":
            return 7
        else:
            return -1

    elif row["Question"] == "SHx Carer Status":
        if row["Answer"] == "Carer for others":
            return 1
        elif row["Answer"] == "Full time carer":
            return 2
        elif row["Answer"] == "No carer":
            return 3
        elif row["Answer"] == "Paid carer":
            return 4
        elif row["Answer"] == "Unpaid/informal carer":
            return 5
        else:
            return -1

    elif row["Question"] == "SHX Lives with":
        if row["Answer"] == "Alone":
            return 1
        elif row["Answer"] == "Carer":
            return 2
        elif row["Answer"] == "Children":
            return 3
        elif row["Answer"] == "Father":
            return 4
        elif row["Answer"] == "Mother":
            return 5
        elif row["Answer"] == "Siblings":
            return 6
        elif row["Answer"] == "Significant other":
            return 7
        elif row["Answer"] == "Spouse":
            return 8
        else:
            return -1
    return row["Answer"]


# %%
df_social_history["Answer2"] = df_social_history.apply(update_answer, axis=1)

# %%
"""
For each encounter ID, Ur, Hospital and Question,
get the max value of the Answer2 column
"""

df_summary = (
    df_social_history.groupby(["Encntr ID", "Ur", "Hospital", "Question", "Answer"])
    .agg(Answer2=("Answer2", "max"))
    .reset_index()
)

# %%
# Drop dulplciates

df_summary = df_summary.drop_duplicates(["Encntr ID", "Ur", "Question"])


# %%
"""
based on them, pviot the data on Question so that the
three columns are added, then drop the Question column
and the values of the columns are from the Answer column

"""

df_pivot_question = df_summary.pivot(
    index=["Encntr ID", "Ur"],
    columns="Question",
    values="Answer",
).reset_index()

df_pivot_numerical = df_summary.pivot(
    index=["Encntr ID", "Ur"],
    columns="Question",
    values="Answer2",
).reset_index()


# %%
"""
do a merge on the index of both the dataframes
"""
df_merged = pd.merge(
    df_pivot_question,
    df_pivot_numerical,
    on=["Encntr ID", "Ur"],
    suffixes=("_question", "_numerical"),
)


# %%
df_merged.to_csv(SAVE_PATH / "social_history_processed.csv", index=False)
# %%

import pandas as pd
import os
from dateutil.relativedelta import relativedelta


dir = os.getcwd()

# Read the data
df = pd.read_excel(dir + '/Training.xlsx', parse_dates=['Completion Date'])

# Sort the DataFrame by Match ID and Completion Date
sorted_df = df.sort_values(by=['Match ID 18Char', 'Completion Date'])

# Save the sorted DataFrame to a new CSV file
sorted_df.to_csv(dir + '/sorted_Training.csv', index=False)

print("Rows have been sorted by Match ID and Completion Date. Output saved to: " + dir + "/sorted_Training.csv.")

df = pd.read_csv(
    dir + "/sorted_Training.csv",
    parse_dates=["Completion Date", "Match Activation Date"], low_memory=False
)

def calculate_months_diff(row):
    """Calculate the difference in months between two dates."""
    if pd.isnull(row["Completion Date"]) or pd.isnull(row["Match Activation Date"]):
        return None  # Handle missing dates
    delta = relativedelta(row["Completion Date"], row["Match Activation Date"])
    return delta.years * 12 + delta.months

# Update the "Match Length" column
df["Match Length"] = df.apply(calculate_months_diff, axis=1)

# Save the updated DataFrame
df.to_csv(dir + "/updated_mathclength_sorted_Training.csv", index=False)

print("Match Length column updated. Output saved to: " + dir +  "/updated_mathclength_sorted_Training.csv.")

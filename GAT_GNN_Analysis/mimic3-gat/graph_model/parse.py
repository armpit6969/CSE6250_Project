import pandas as pd
import glob
import os

# Define the directory containing the CSV files
directory = os.path.dirname(os.path.abspath(__file__))

# Initialize a list to store results
results = []

# Iterate through each CSV file in the directory
for filepath in glob.glob(os.path.join(directory, '*.csv')):
    df = pd.read_csv(filepath)
    # Select the last row
    last_row = df.tail(1)
    if not last_row.empty:
        wt_auc = last_row['wt_auc'].values[0]
        micro_auc = last_row['micro_auc'].values[0]
        macro_auc = last_row['macro_auc'].values[0]
        results.append({
            'file': os.path.basename(filepath),
            'wt_auc': wt_auc,
            'micro_auc': micro_auc,
            'macro_auc': macro_auc
        })

# Create a DataFrame from the results
combined_df = pd.DataFrame(results)

# Save the combined results to a new CSV file
combined_df.to_csv('combined_metrics_last_epoch.csv', index=False)
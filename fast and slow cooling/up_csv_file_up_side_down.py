import os
import pandas as pd

def reverse_rows_in_csv_files(folder_path):
    # Iterate through files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV and includes "up" in its name
        if filename.endswith(".csv") and "up" in filename.lower():
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Reverse all rows
            reversed_df = df.iloc[::-1]
            
            # Save the modified dataframe back to the original file
            reversed_df.to_csv(file_path, index=False)

# Example usage
folder_path = '/workspaces/kakalioslab/fast and slow cooling/fast and slow cool/fast and slow cool fixed time /slow cool'  # Replace with your folder path
reverse_rows_in_csv_files(folder_path)

import pandas as pd
import os

# 獲取當前腳本所在的目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the original data
file_path = os.path.join(script_dir, '../Excel/Processed_Circle_test.xlsx')
sheet_data = pd.read_excel(file_path, sheet_name='Sheet1')

# Create copies for test and train datasets
test_data = sheet_data.copy()
train_data = sheet_data.copy()

# Process test_data: retain the first non-null value in each group of 5, replace others with 'X', but do not change the first column
num_rows, num_cols = sheet_data.shape
for col in sheet_data.columns:
    if col != sheet_data.columns[0]:  # Skip the first column if it's not part of the data to be changed
        for i in range(0, num_rows, 5):
            first_found = False
            for j in range(5):
                if i + j < num_rows:
                    if not first_found and pd.notna(sheet_data.at[i + j, col]):
                        first_found = True
                    else:
                        test_data.at[i + j, col] = 'X'
            # If no non-null value found in the group, keep the column as is
            if not first_found:
                for j in range(5):
                    if i + j < num_rows:
                        test_data.at[i + j, col] = 'X'

# Process train_data: replace all blank cells with 'X'
train_data = train_data.fillna('X')

# Replace values in the train data that are retained in the test data with 'X'
for col in test_data.columns:
    if col != sheet_data.columns[0]:  # Skip the first column if it's not part of the data to be changed
        for i in range(num_rows):
            if test_data.at[i, col] != 'X':
                train_data.at[i, col] = 'X'

# Save the processed test data into a new Excel file
test_file_path = os.path.join(script_dir, '../Excel/Processed_Circle_test_test.xlsx')
test_data.to_excel(test_file_path, index=False)

# Save the processed train data into a new Excel file
train_file_path = os.path.join(script_dir, '../Excel/Processed_Circle_test_train.xlsx')
train_data.to_excel(train_file_path, index=False)

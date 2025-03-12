import pandas as pd

def compare_csvs(file1, file2):
    # Load CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure both files have the same columns
    if set(df1.columns) != set(df2.columns):
        print("Files have different columns!")

    # Sort by index to ensure proper comparison
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    # Calculate percentage of matching rows
    total_rows = len(df2)
    matching_rows = (df1 == df2).all(axis=1).sum()
    match_percentage = (matching_rows / total_rows) * 100

    # Print results
    print('Percentage of file2 that matches file1: {}'.format(match_percentage))

if __name__ == '__main__':        
    compare_csvs('./exp2_aug1/submission.csv', './exp2_aug1/submission_old.csv')
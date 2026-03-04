

import pandas as pd
import numpy as np
import os

# Base directory — always points to where data_processing.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

def clean_data(df):
    # Drop irrelevant data
    data = df.drop(columns=['UserID'])
    
    # Save to data folder regardless of where function is called from
    output_path = os.path.join(DATA_DIR, 'final_version.csv')
    # data.to_csv(output_path, index=False)
    data.to_csv('../data/final_version',index=False)
    # print(f'Clean training data saved to: {output_path}')
    print('Clean training data saved')
    
    return data

def clean_data_test(df):
    # Drop irrelevant data
    data = df.drop(columns=['UserID','ordered'])
    
    # Save to data folder
    output_path = os.path.join(DATA_DIR, 'final_test_version.csv')
    data.to_csv(output_path, index=False)
    print(f'Clean test data saved to: {output_path}')
    
    return data

if __name__ == '__main__':
    # Training data
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'training_sample.csv'))
    clean_data(df_train)

    # Test data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'testing_sample.csv'))
    clean_data_test(df_test)
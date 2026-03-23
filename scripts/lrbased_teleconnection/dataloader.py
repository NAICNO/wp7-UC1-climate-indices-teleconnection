# dataloader.py

import pandas as pd
import numpy as np

def load_data(data_file, delete_features, year_start=None):
    data = pd.read_csv(data_file)
    if year_start is not None:
        data = data[data['Time'] >= year_start]
    for delete_feature in delete_features:
        del data[delete_feature]
    data.reset_index(drop=True, inplace=True)
    return data

def preprocess_data(data, target_feature):
    cols = ['Time', target_feature] + [col for col in data.columns if col not in ['Time', target_feature]]

    data = data[cols].copy()

    # Get numeric columns (all except Time)
    numeric_cols = data.columns[1:]

    # Convert numeric columns to float to avoid pandas dtype errors
    for col in numeric_cols:
        data[col] = data[col].astype(float)

    # Min-max normalization to 0-100 range
    for col in numeric_cols:
        col_min = data[col].min()
        col_max = data[col].max()
        if col_max != col_min:
            data[col] = (data[col] - col_min) / (col_max - col_min) * 100
        else:
            data[col] = 0.0

    return data.dropna()

def generate_lagdata(init_lag, max_lag, data, target_feature, with_mean_feature=False):
    columnnames = data.columns[1:]
    lagged_data = [data]

    for lag in range(init_lag, max_lag):
        lagged_columns = data[columnnames].shift(lag)
        lagged_columns.columns = [f'{lag}_lag_{col}' for col in columnnames]
        lagged_data.append(lagged_columns)
        
    data = pd.concat(lagged_data, axis=1)
    columns_to_drop = [col for col in columnnames if col != target_feature]
    data.drop(columns=columns_to_drop, inplace=True)
    
    if(with_mean_feature and init_lag+1< max_lag-1):      
        
        mean_data = [data.copy()]

        for col in columnnames: 
            statscolumn = []
            for lag in range(init_lag+1, max_lag):
                statscolumn.append(f'{lag}_lag_{col}')
                
                # Calculate mean, median, std for the lagged DataFrame
                mean_values = data[statscolumn].mean(axis=1)
                median_values = data[statscolumn].median(axis=1)
                std_values = data[statscolumn].std(axis=1)
                
                # Append mean, median, std columns with appropriate names
                mean_df = pd.DataFrame(mean_values, columns=[f'{init_lag}_lag_{lag}_mean_{col}'])
                median_df = pd.DataFrame(median_values, columns=[f'{init_lag}_lag_{lag}_median_{col}'])
                std_df = pd.DataFrame(std_values, columns=[f'{init_lag}_lag_{lag}_std_{col}'])
                
                #mean_data.append(mean_df)
                mean_data.append(median_df)
                #mean_data.append(std_df)
                
            
        data = pd.concat(mean_data, axis=1)
        
        data = data.dropna(axis=1, how='all')
        # Optionally, convert column names to string type
        data.columns = data.columns.astype(str)


    return data.dropna()

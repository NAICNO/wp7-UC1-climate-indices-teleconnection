# main.py

import argparse
import sys
import os
import pandas as pd
from IPython.display import clear_output  # For Jupyter clean-up

# Add the scripts directory to the system path
sys.path.append(os.path.dirname(__file__))
from libs import *
from dataloader import load_data, preprocess_data, generate_lagdata
from models import ModelSelector
from plotting import plot_mae_vs_corr_reporting, plot_climate_index_analysis
from evaluation import evaluate_and_plot_model


def main(data_file, target_feature, delete_features, modelname, max_allowed_features, end_lag, n_ensembles, splitsize=0.6, with_mean_feature=False, main_year_start=None, step_lag=10, with_wavelet_filter=False, is_jupyter_run=False):
    data = load_data(data_file, delete_features, year_start=main_year_start).dropna()
    
    # Preprocessing data and initializing results
    data = preprocess_data(data, target_feature)
    logged_results = {}
    init_lag = 15
    
    if is_jupyter_run:
        # Clear the previous output for cleaner display
        clear_output(wait=True)
        # Add a descriptive message about the current run
        print(f"Running model: {modelname}\n"
              f"Target feature: {target_feature}\n"
              f"Data file: {data_file}\n"
              f"Max allowed features: {max_allowed_features}\n"
              f"Step lag: {step_lag}\n"
              f"Split size: {splitsize}\n"
              f"Ensembles: {n_ensembles}\n"
              f"Wavelet filtering: {with_wavelet_filter}\n")
    
    # Iterating through lags and running the model
    for num_irrelevant, max_lag in tqdm(enumerate(range(init_lag, end_lag + 1)), desc="Processing lag steps"):
        # Clear previous loop outputs (for Jupyter)
        if is_jupyter_run:
            if(num_irrelevant % 4 == 0):
                clear_output(wait=True)
                print(f"Processing lag step {num_irrelevant + 1}...")
                print(f"Current max lag: {max_lag}, Initial lag: {init_lag}")

        # Generate lagged data
        init_lag = max(1, max_lag - step_lag)
        data_lagged = generate_lagdata(init_lag, max_lag, data, target_feature, with_mean_feature)
        
        print(f"Step {num_irrelevant}: Generated lagged data with {len(data_lagged.columns)} features "
              f"from {len(data_lagged)} rows (original data size: {len(data)} rows)")
        
        # Extracting features and labels
        Time_data = data_lagged["Time"].copy()
        X = data_lagged.drop(['Time', target_feature], axis=1)
        y = data_lagged[target_feature]

        # Splitting the dataset
        split_index = int(len(X) * splitsize)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
            
        if with_wavelet_filter: 
            X_train = wavelet_filter_dataframe(X_train, 0) 
            X_test = wavelet_filter_dataframe(X_test, 0) 
            y_train = wavelet_filter_dataframe(y_train, 1) 
            y_test = wavelet_filter_dataframe(y_test, 1) 

        # Initialize feature importance tracking
        feature_importances = np.zeros(X_train.shape[1])

        for _ in range(n_ensembles):
            model = ModelSelector(modelname)
            model.fit(X_train, y_train)
            feature_importances += model.feature_importances()

        feature_importances /= n_ensembles
        selected_indices = np.argsort(feature_importances)[-max_allowed_features:]
        selected_features = X.columns[selected_indices]

        # Select and train model with top features
        X_train_selected = X_train.iloc[:, selected_indices]
        X_test_selected = X_test.iloc[:, selected_indices]

        model_selected = ModelSelector(modelname)
        model_selected.fit(X_train_selected, y_train)
        y_pred = model_selected.predict(X_test_selected)

        # Evaluate model
        mae = mean_absolute_error(y_test, y_pred)
        corr_score, _ = pearsonr(y_test, y_pred)
        max_forecast = min([int(fname.split("_lag_")[0]) for fname in selected_features.tolist()])

        dataset_name = data_file.replace("dataset/noresm-f-p1000_", "").replace("_new_jfm.csv", "")
        logged_results[f"{modelname}_{num_irrelevant}"] = {
            "model": modelname,
            "num_irrelevant": num_irrelevant,
            "max_allowed_features": max_allowed_features,
            "target_feature": target_feature,
            "dataset": dataset_name,
            "year_start": min(Time_data),
            "year_end": max(Time_data),
            "splitsize": splitsize,
            "max_lag": max_lag,
            "init_lag": init_lag,
            "max_forecast": max_forecast,
            "mae_score": mae,
            "corr_score": corr_score,
            "n_total_features": len(X.columns),
            "waveletfilter": with_wavelet_filter,
            "selected_feature_list": selected_features.tolist(),
            "feature_importances": model_selected.feature_importances().tolist(),
            "selected_features": ", ".join(selected_features.tolist()) + f"_{len(X.columns)}_{max_forecast}"
        }

        # Output selected features
        print(f"Selected Features: {selected_features.tolist()} (from {len(X.columns)} total features)")

        # Generate filename and save results
        filename = f'{target_feature}_{max_allowed_features}_{splitsize}_{step_lag}_results.png'
        if len(delete_features) > 1:
            filename = f'del{len(delete_features)}_{filename}'
        if with_wavelet_filter:
            filename = f'waveletfilter_{filename}'

        # Create the directory path
        dirpath = os.path.join("results", "gen3.0", dataset_name, target_feature, modelname, str(min(data["Time"])), str(max(data["Time"])))
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        
        # Save results to CSV
        logged_results_df = pd.DataFrame.from_dict(logged_results, orient='index')
        logged_results_df.to_csv(filepath[:-3] + 'csv', index=False)

        # Print status for each iteration
        print(f"Iteration {num_irrelevant + 1}: MAE = {mae:.4f}, Correlation = {corr_score:.4f}")
        print(f"Results saved to {filepath[:-3]}csv\n")
      
    if is_jupyter_run:  
        return filepath[:-3] + 'csv'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='dataset/noresm-f-p1000_shigh_new_jfm.csv', help="Path to the data file")
    parser.add_argument("--target_feature", type=str, default="amoSSTmjjaso", help="Target feature for forecasting")
    parser.add_argument("--delete_features", nargs='+', default=["amoSSTjfm"], help="List of features to delete")
    parser.add_argument("--modelname", type=str, required=True, help="Name of the model")
    parser.add_argument("--max_allowed_features", type=int, default=6, help="Maximum number of features to select")
    parser.add_argument("--end_lag", type=int, default=100, help="End lag forecast")
    parser.add_argument("--n_ensembles", type=int, default=100, help="Number of ensembles")
    parser.add_argument("--main_year_start", type=int, default=850, help="Year start to filter data")
    parser.add_argument("--splitsize", type=float, default=0.6, help="Split size for train-test split")
    parser.add_argument("--with_mean_feature", action='store_true', help="Include mean feature in the analysis")
    parser.add_argument("--with_wavelet_filter", action='store_true', help="Perform wavelet filtering")
    parser.add_argument("--step_lag", type=int, default=10, help="Maximum lag range")

    args = parser.parse_args()
    n_ensembles = args.n_ensembles

    print(args)

    if args.modelname == "LinearRegression":  # Deterministic model does not need multiple runs
        n_ensembles = 1

    delete_features = str(args.delete_features[0]).split(",")
    print(f"Deleted features: {delete_features}")

    main(args.data_file,
         args.target_feature,
         delete_features,
         args.modelname,
         args.max_allowed_features,
         args.end_lag,
         n_ensembles,
         splitsize=args.splitsize,
         with_mean_feature=args.with_mean_feature,
         main_year_start=args.main_year_start,
         step_lag=args.step_lag,
         with_wavelet_filter=args.with_wavelet_filter)

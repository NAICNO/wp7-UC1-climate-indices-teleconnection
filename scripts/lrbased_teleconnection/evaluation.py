from libs import *
def plot_scatter_plot(ax, y_test, y_pred, labels):
    """
    Plot a scatter plot of actual vs predicted values on the given axis.
    
    Parameters:
    - ax (matplotlib.axes.Axes): The subplot axis to plot on.
    - y_test (pd.Series): Actual target values for testing data.
    - y_pred (np.array): Predicted values from the model.
    """
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', color='red')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='black')
    ax.set_xlabel(labels["actual"][0]+' Values')
    ax.set_ylabel(labels["predicted"][0]+' Values')
    #ax.set_title(labels["actual"][0]+' vs. '+labels["predicted"][0])
    
    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', 'box')
    
    # Set the same limits for x and y axis
    max_range = max(y_test.max(), y_pred.max())
    min_range = min(y_test.min(), y_pred.min())
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    
def evaluate_and_plot_model(filtered_df, target_feature, delete_features, generate_lagdata, load_data, preprocess_data, ModelSelector, ax, waveletfilters=[True, False], max_features_forced=8, otheraxes=None):
    """
    Evaluate and plot model performance for a specific target feature.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing evaluation metrics.
    - target_feature (str): The target feature to evaluate.
    - delete_features (list): List of features to be removed from the dataset.
    - generate_lagdata (function): Function to generate lagged data.
    - load_data (function): Function to load data from a file.
    - preprocess_data (function): Function to preprocess data.
    - ModelSelector (class): Class to create the model instance.
    - ax (matplotlib.axes.Axes): The subplot axis to plot on.
    - max_features_forced (int): Maximum number of features to be considered.
    
    Returns:
    - info_text (str): Information text containing evaluation metrics.
    """
    # Obtain the best model parameters
    lowest_mae_score = filtered_df[filtered_df["target_feature"] == target_feature]['mae_score'].idxmin()
    lowest_mae_score_row = filtered_df.loc[lowest_mae_score]
    data_file = f"dataset/noresm-f-p1000_{str(lowest_mae_score_row['dataset'])}_new_jfm.csv"

    # Extract parameters from the row with the lowest MAE score
    dataset = str(lowest_mae_score_row['dataset'])
    modelname = lowest_mae_score_row["model"]
    splitsize = lowest_mae_score_row["splitsize"]
    logged_corr_score = lowest_mae_score_row["corr_score"]
    year_start = lowest_mae_score_row["year_start"]
    max_allowed_features = lowest_mae_score_row["max_allowed_features"]
    logged_mae = lowest_mae_score_row["mae_score"]
    max_lag = lowest_mae_score_row["max_lag"]
    is_amo_guided = bool(lowest_mae_score_row["is_amo_guided"])
    init_lag = lowest_mae_score_row["init_lag"]
    max_forecast = lowest_mae_score_row["max_forecast"]
    selected_indices = eval(lowest_mae_score_row["selected_feature_list"])[:max_features_forced]
    logged_feature_importances = eval(lowest_mae_score_row["feature_importances"])[:max_features_forced]
    labelmapping = {
        False: {
            "actual": [
                "Actual", "skyblue"  # Smooth light blue
            ],
            "predicted": [
                "Predicted", "orange"  # Contrasts well with skyblue
            ],
        },
        
        True: {
            "actual": [
                "Actual (with wave filtering)", "midnightblue"  # Dark blue for contrast
            ],
            "predicted": [
                "Predicted (with wave filtering)", "coral"  # Contrasts well with midnightblue
            ],
        }
    }


    axes = [ax]
    
    for waveletfilter in waveletfilters:
        dict_feature_importances = {}

        # Load and preprocess data
        data = load_data(data_file, delete_features, year_start=year_start).dropna()

            
        data = preprocess_data(data, target_feature)

        # Process feature importances
        unique_features = {}
        for idx, selected_index in enumerate(selected_indices):
            dict_feature_importances[selected_index] = float(logged_feature_importances[idx])
            feature_name = selected_index.split("_")[-1]
            if feature_name not in unique_features:
                unique_features[feature_name] = 0
            unique_features[feature_name] += 1

        # Generate lagged data
        data_lagged = generate_lagdata(init_lag, max_lag, data, target_feature, with_mean_feature=True)
        TimeAxis = data_lagged["Time"].copy()
        X = data_lagged.drop(['Time', target_feature], axis=1)
        y = data_lagged[target_feature]

        # Split data into training and testing sets
        split_index = int(len(X) * splitsize)
        X_train_selected, X_test_selected = X.iloc[:split_index][selected_indices], X.iloc[split_index:][selected_indices]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


        if(waveletfilter): 
            freq_band = [50., 70.]
            X_train_selected = wavelet_filter_dataframe(X_train_selected, 0, freq_band=freq_band) 
            X_test_selected = wavelet_filter_dataframe(X_test_selected, 0, freq_band=freq_band) 
            y_train = wavelet_filter_dataframe(y_train, 1, freq_band=freq_band) 
            y_test = wavelet_filter_dataframe(y_test, 1, freq_band=freq_band) 
                
                
        # Initialize and train the model
        model_selected = ModelSelector(modelname)
        if modelname in ["LRforcedPSO"]:
            model_selected.set_coef(logged_feature_importances, X_train_selected, y_train)
        else:
            model_selected.fit(X_train_selected, y_train)
            if modelname in ["LinearRegression"]:
                model_selected.coef_ = np.array(logged_feature_importances)

        # Make predictions and evaluate the model
        y_pred = model_selected.predict(X_test_selected)
        mae = mean_absolute_error(y_test, y_pred)
        corr_score, _ = pearsonr(y_test, y_pred)
        
        labelmapping[waveletfilter]["mae"] = mae
        labelmapping[waveletfilter]["corr_score"] = corr_score

        # Plot actual vs predicted values
        ax.plot(TimeAxis[split_index:], y_test, label=labelmapping[waveletfilter]["actual"][0], color=labelmapping[waveletfilter]["actual"][1])
        ax.plot(TimeAxis[split_index:], y_pred, label=labelmapping[waveletfilter]["predicted"][0], color=labelmapping[waveletfilter]["predicted"][1])
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

        # Plot residuals
        residuals = y_test - y_pred
        ax2 = ax.twinx()
        #ax2.plot(TimeAxis[split_index:], residuals, color='orange', label='Residuals')
        #ax2.set_ylabel('Residuals', color='orange')
        #ax2.tick_params(axis='y', labelcolor='orange')

        # Plot confidence intervals (dummy example)
        pred_intervals = np.array([5] * len(y_pred))  # Replace with actual intervals
        upper_bound = y_pred + pred_intervals
        lower_bound = y_pred - pred_intervals

        if waveletfilter == False:
            ax.fill_between(TimeAxis[split_index:], lower_bound, upper_bound, color='grey', alpha=0.15, label='Confidence Interval')

        # Plot Actual vs Predicted scatter plot
        #fig, ax4 = plt.subplots(figsize=(12, 6))
        #ax4.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', color='red')
        #ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='black')
        #ax4.set_xlabel('Actual Values')
        #ax4.set_ylabel('Predicted Values')
        #ax4.set_title('Actual vs. Predicted')
        #plt.show()

        # Plot the scatter plot #[x0, y0, width, height]
        
        if(False in waveletfilters):
            if waveletfilter: 
                scatter_ax = ax.inset_axes([0.52, -0.8, 0.7, 0.7])  # Adjusted upwards more and resized smaller
            else:
                scatter_ax = ax.inset_axes([0.2, -0.8, 0.7, 0.7])  # Adjusted upwards more and resized smaller
                axes.append(scatter_ax)

            plot_scatter_plot(scatter_ax, y_test, y_pred, labelmapping[waveletfilter])

        
    # Set plot title and information text
    feature_names = textwrap.fill(', '.join(unique_features.keys()), width=140)
    # Collect legends from all axes
    handles, labels = [], []
    for ax_ in axes:
        for handle, label in zip(*ax_.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    # Add legend to the main axis
    ax.legend(handles, labels, loc='upper left')
    
    ax.set_title(f'{target_feature} forecasts for {max_forecast} years using {modelname} on {min(max_allowed_features, max_features_forced)} features from {dataset} dataset \n({len(unique_features)} uniques: {feature_names})')
    
    if  "mae" in labelmapping[False]:            
        info_text = f'MAE: {labelmapping[False]["mae"]:.2f}\nCorrelation: {labelmapping[False]["corr_score"]:.2f}'
        if is_amo_guided:
            info_text = f'**MAE: {labelmapping[False]["mae"]:.2f}\nCorrelation: {labelmapping[False]["corr_score"]:.2f}'
    else:
        info_text = "\n"
        
    info_text = f'{info_text}\n Logged-MAE: {logged_mae:.2f}\nLogged-Correlation: {logged_corr_score:.2f}\n----------------'
        
    if  "mae" in labelmapping[True]:
        info_text = f'{info_text}\n MAE (with filter): {labelmapping[True]["mae"]:.2f}\nCorrelation (with filter): {labelmapping[True]["corr_score"]:.2f}\n----------------'
    
    feature_info = '\n'.join([f'{feature}: {float(score):.4f}, {float(dict_feature_importances[feature]):.4f}' for feature, score in sorted(zip(selected_indices, model_selected.feature_importances()), key=lambda x: x[1], reverse=True)])
    info_text += f'\n\nFeature Importances:\n{feature_info}'
    
    ax.grid(True)
    
    return info_text

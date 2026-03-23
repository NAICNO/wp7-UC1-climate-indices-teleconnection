import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from evaluation import evaluate_and_plot_model


def plot_mae_vs_corr_reporting(filtered_df, target_feature, width=15, ax=None):
    """
    Plot MAE vs Pearson Correlation score for a specific target feature.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing evaluation metrics.
    - target_feature (str): The target feature to evaluate.
    - width (int): Width for wrapping text labels.
    - ax (matplotlib.axes.Axes): The subplot axis to plot on.
    """
    # Filter and sort the DataFrame
    targetfilter_df = filtered_df[filtered_df["target_feature"] == target_feature]
    targetfilter_df = targetfilter_df.sort_values(by='mae_score')
    targetfilter_df = targetfilter_df.drop_duplicates(subset='max_forecast').head(8)

    def wrap_labels(labels, width):
        out_labels = []
        for idx, out_label in enumerate(labels):
            label = out_label[:-(2+len(out_label.split("_")[-1])+len(out_label.split("_")[-2]))].split(",") 
            out_labels.append(f'fcast: {out_label.split("_")[-1]}, n_feature: {out_label.split("_")[-2]}' + '\n' + '\n'.join(label[:6]))
            if(len(label) > 6):
                out_labels[idx] = out_labels[idx] + '\n.....'
        return out_labels

    x_order = range(1, len(targetfilter_df) + 1)
    ax.plot(x_order, targetfilter_df["mae_score"], color='blue')
    ax.set_xlabel("Selected Features")
    ax.set_ylabel("MAE Score", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    ax2 = ax.twinx()
    ax2.plot(x_order, targetfilter_df["corr_score"], color='red')
    ax2.set_ylabel("Pearson Score", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.invert_yaxis()

    wrapped_labels = wrap_labels(targetfilter_df["selected_features"], width)
    ax.set_xticks(x_order)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='right')
    ax.grid(True)
    ax.set_title(f"Pearson Correlation and MAE Score with lagged predictor selection on {target_feature}")

def combined_plots(filtered_df, target_feature, delete_features, generate_lagdata, load_data, preprocess_data, ModelSelector, waveletfilters=[True, False]):
    """
    Generate and display combined plots for model evaluation and MAE vs correlation.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing evaluation metrics.
    - target_feature (str): The target feature to evaluate.
    - delete_features (list): List of features to be removed from the dataset.
    - generate_lagdata (function): Function to generate lagged data.
    - load_data (function): Function to load data from a file.
    - preprocess_data (function): Function to preprocess data.
    - ModelSelector (class): Class to create the model instance.
    """
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Evaluate and plot model performance on the first subplot
    info_text = evaluate_and_plot_model(filtered_df, target_feature, delete_features, generate_lagdata, load_data, preprocess_data, ModelSelector, ax1, waveletfilters=waveletfilters)
    
    fig.text(0.34, 0.43, info_text, horizontalalignment='right', verticalalignment='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Plot MAE vs Pearson Correlation score on the third subplot
    plot_mae_vs_corr_reporting(filtered_df, target_feature, width=15, ax=ax2)

    plt.tight_layout()
    # Adjust the position of ax2 lower by -0.2 in relative coordinates
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, 0+0.22, pos2.width, pos2.height-0.05])
    plt.show()


def plot_climate_index_analysis(filtered_df, corr_score=0.6, mae_score=14):
    """
    Plot climate index teleconnection analysis results.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing evaluation metrics.
    - corr_score (float): Minimum correlation score for plotting.
    - mae_score (float): Maximum MAE score for plotting.
    """
    figsize = 0.6 * len(filtered_df["target_feature"].unique())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, min(16, figsize)))  # Create a 2x2 grid of subplots

    fig.suptitle("Climate index teleconnection analysis results", fontsize=16)

    # Plot max_forecast vs target_feature vs corr_score
    sc1 = ax1.scatter(filtered_df["max_forecast"], filtered_df["target_feature"], c=filtered_df["corr_score"], vmin=0, vmax=0.9)
    ax1.set_xlabel("max_forecast")
    ax1.set_ylabel("target_feature")
    ax1.grid(True)

    # Plot mae_score vs target_feature vs corr_score
    sc2 = ax2.scatter(filtered_df["mae_score"], filtered_df["target_feature"], c=filtered_df["corr_score"], vmin=0, vmax=0.9)
    ax2.set_xlabel("mae_score")
    ax2.set_ylabel("target_feature")
    ax2.grid(True)
    ax2.set_xlim(5, mae_score + 0.1)
    ax2.invert_xaxis()  # Invert x-axis

    # Plot max_forecast vs target_feature vs mae_score
    sc3 = ax3.scatter(filtered_df["max_forecast"], filtered_df["target_feature"], c=filtered_df["mae_score"], vmin=5, vmax=15.0, cmap='viridis_r')
    ax3.set_xlabel("max_forecast")
    ax3.set_ylabel("target_feature")
    ax3.grid(True)

    # Plot corr_score vs target_feature vs mae_score
    sc4 = ax4.scatter(filtered_df["corr_score"], filtered_df["target_feature"], c=filtered_df["mae_score"], vmin=5, vmax=15.0, cmap='viridis_r')
    ax4.set_xlabel("corr_score")
    ax4.set_ylabel("target_feature")
    ax4.grid(True)
    ax4.set_xlim(corr_score - 0.01, 1)

    # Create colorbars for scatter plots
    divider1 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="5%", pad=0.6)
    cbar1 = fig.colorbar(sc1, cax=cax1, orientation='vertical')
    cbar1.set_label("corr_score")

    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes("right", size="5%", pad=0.6)
    cbar2 = fig.colorbar(sc3, cax=cax2, orientation='vertical')
    cbar2.set_label("mae_score")

    plt.tight_layout()
    plt.show()

def plot_mae_corr(num_irrelevant, logged_results, modelname, filename, rotate=False, year_length=300):
    """
    Plot MAE and Pearson Correlation score against the number of irrelevant features.

    Parameters:
    - num_irrelevant (int): Number of irrelevant features to plot.
    - logged_results (dict): Dictionary containing logged results for different numbers of irrelevant features.
    - modelname (str): Name of the model used.
    - filename (str): Filename to save the plot.
    - rotate (bool): Whether to rotate the plot for better visibility.
    - year_length (int): Year length for windowing MAE and correlation scores.
    """
    num_irrelevants = range(1, num_irrelevant)
    selected_features = [logged_results[f"{modelname}_{num}"]["selected_features"] for num in num_irrelevants]
    num_irrelevant_range = [logged_results[f"{modelname}_{num}"]["num_irrelevant"] for num in num_irrelevants]
    mae_scores = [logged_results[f"{modelname}_{num}"]["mae_score"] for num in num_irrelevants]
    corr_scores = [logged_results[f"{modelname}_{num}"]["corr_score"] for num in num_irrelevants]

    windowing_maescore = [np.median(logged_results[f"{modelname}_{num}"][str(year_length) + "_windowing_maescore"]) for num in num_irrelevants]
    windowing_coorscore = [np.median(logged_results[f"{modelname}_{num}"][str(year_length) + "_windowing_coorscore"]) for num in num_irrelevants]

    figsize = (8, 12) if rotate else (12, 8)
    filename = "rotated_" + filename if rotate else filename
    fig, ax1 = plt.subplots(figsize=figsize)

    if rotate:
        ax1.set_yticks(num_irrelevant_range)
        ax1.set_yticklabels(selected_features)
        ax1.set_ylabel('Number of Irrelevant Features')
        ax1.set_xlabel('Mean Absolute Error', color='tab:blue')
        ax1.plot(mae_scores, num_irrelevant_range, color='tab:blue', label='MAE')
        ax1.plot(windowing_maescore, num_irrelevant_range, color='#ADD8E6', label='Windowing MAE')
        ax1.tick_params(axis='x', labelcolor='tab:blue')

        ax2 = ax1.twiny()
        ax2.set_xlabel('Pearson Correlation Score', color='tab:red')
        ax2.plot(corr_scores, num_irrelevant_range, color='tab:red', label='Correlation')
        ax2.plot(windowing_coorscore, num_irrelevant_range, color='#FFB6C1', label='Windowing Correlation')
        ax2.tick_params(axis='x', labelcolor='tab:red')
        ax2.invert_xaxis()
    else:
        logged_results_df = pd.DataFrame.from_dict(logged_results, orient='index')
        logged_results_df.to_csv(filename[:-3] + 'csv', index=False)
        ax1.set_xticks(num_irrelevant_range)
        ax1.set_xticklabels(selected_features, rotation=90)
        ax1.set_xlabel('Number of Irrelevant Features')
        ax1.set_ylabel('Mean Absolute Error', color='tab:blue')
        ax1.plot(num_irrelevant_range, mae_scores, color='tab:blue', label='MAE')
        ax1.plot(num_irrelevant_range, windowing_maescore, color='#ADD8E6', label='Windowing MAE')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Pearson Correlation Score', color='tab:red')
        ax2.plot(num_irrelevant_range, corr_scores, color='tab:red', label='Correlation')
        ax2.plot(num_irrelevant_range, windowing_coorscore, color='#FFB6C1', label='Windowing Correlation')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.invert_yaxis()

    ax1.grid(True)
    ax2.grid(True)
    
    # Adjust margins for better layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    
    plt.title(f'Optimal MAE and Pearson Correlation on lagged data (suitable for forecasting) using {modelname}')

    legend_labels = ['MAE', 'Correlation', 'Windowing MAE', 'Windowing Correlation']
    legend_colors = ['tab:blue', 'tab:red', '#ADD8E6', '#FFB6C1']
    legend_handles = [plt.Line2D([0], [0], color=color, linewidth=3) for color in legend_colors]
    plt.legend(legend_handles, legend_labels, loc='upper left')

    plt.savefig(filename)
    plt.show()
    plt.close()

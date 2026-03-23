# combined_plots.py

import matplotlib.pyplot as plt
from evaluation import evaluate_and_plot_model
from plotting import plot_mae_vs_corr_reporting

def combined_plots(filtered_df, target_feature, delete_features, generate_lagdata, load_data, preprocess_data, ModelSelector):
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))

    info_text = evaluate_and_plot_model(filtered_df, target_feature, delete_features, generate_lagdata, load_data, preprocess_data, ModelSelector, axs[0])
    plot_mae_vs_corr_reporting(filtered_df, target_feature, width=15, ax=axs[1])
    fig.text(0.11, 0.55, info_text, horizontalalignment='right', verticalalignment='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.show()

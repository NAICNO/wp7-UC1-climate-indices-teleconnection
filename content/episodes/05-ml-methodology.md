# Machine Learning Methodology

```{objectives}
- Understand the data preparation process
- Learn about the available ML models
- Know how results are analyzed
```

## Methodology

### Data Preparation

1. **Data Collection:** Data on 65 climate indices from NORCE Climate and Environment datasets (1845-2005) is used. Additional datasets can be added to the `dataset` directory.
2. **Preprocessing:** Data is normalized to a uniform scale (0-100) to ensure consistency across indices.
3. **Dataset Splitting:** The data is split into training (60%) and testing (40%) sets, ensuring the model is trained on historical data without information leaks.
4. **Feature Engineering:** Lagged features are introduced to capture temporal dependencies, and median values are used to handle noise and errors in the data.

### Model Selection and Training

Several machine learning models are employed to predict climate indices:

- **Linear Regression (LR):** A simple baseline model for linear relationships.
- **Random Forest Regressor:** An ensemble model that improves accuracy and reduces overfitting by combining multiple decision trees.
- **Multi-Layer Perceptron (MLP) Regressor:** A neural network model capable of capturing complex, non-linear relationships.
- **XGBoost Regressor:** A gradient boosting algorithm optimized for both speed and performance, with GPU support to accelerate training.

### VM Deployment

Handling large datasets and multiple models benefits from computational resources available on NAIC Orchestrator VMs:

- **GPU Acceleration:** XGBoost and MLP models can leverage GPU acceleration when available on the VM.
- **Background Processing:** Long-running experiments can be executed in tmux sessions to persist beyond SSH disconnection.
- **Interactive Analysis:** Jupyter Lab provides an interactive environment for exploring results and running experiments.

### Results Analysis

- **Performance Metrics:** Models are evaluated using correlation coefficients for linear relationships and MAE for accuracy.
- **Visualization:** Accuracy landscapes and feature importance plots are generated to interpret the model's performance and identify key predictors.
- **Collaboration with Climate Scientists:** Domain knowledge is incorporated to refine models and interpret results meaningfully.

## Practical Implementation

### Search Methodology

The main methodology involves systematically testing various ML models on climate data. The process includes generating lagged features, splitting data, applying wavelet filtering (optional), and ranking features by importance. This is done iteratively across different model configurations to find the best-performing combination.

### Flowchart

Here's a simplified flowchart of the overall process:

```{mermaid}
graph TD
    A[Start] --> B[Load Data]
    B --> C[Filter & Preprocess Data]
    C --> D[Generate Lagged Data]
    D --> E[Train Model]
    E --> F[Evaluate Model]
    F --> G[Log Results]
    G --> H[Generate Visualizations]
    H --> I[End]
```

### Data Analysis

The data analysis methodology involves loading evaluation results, calculating performance metrics, and aggregating results to identify the best-performing model configurations. Visualization tools are used to provide insights into model performance and key predictors.

```{keypoints}
- Data is normalized to 0-100 scale and split 60/40 for training/testing
- Four ML models are available: Linear Regression, Random Forest, MLP, XGBoost
- Lagged features capture temporal dependencies between indices
- Models are evaluated using correlation coefficients and MAE
```

# NAIC Teleconnection Demonstrator FAQ

Welcome to the comprehensive FAQ for the NAIC Teleconnection Demonstrator, an innovative Jupyter-based framework that integrates machine learning (ML) to analyze climate teleconnections, with a focus on Atlantic Multidecadal Variability (AMV) and Pacific Decadal Variability (PDV). This document is organized into five categories to guide climate scientists, data scientists, policy makers, and enthusiasts in understanding, setting up, and leveraging this powerful tool for advanced climate research.

## Category A: Understanding the Demonstrator

### 1. What is the NAIC Teleconnection Demonstrator and what problem does it solve?
The NAIC Teleconnection Demonstrator is a platform that combines machine learning to uncover teleconnections—statistical relationships between climate indices like the Atlantic Multidecadal Oscillation (AMO) and North Atlantic Oscillation (NAO). Traditional climate models often struggle with the non-linear complexity of these relationships, which are critical for predicting long-term climate trends like AMV and PDV. By analyzing 65 climate indices across centuries of data, the demonstrator delivers precise, decadal-scale forecasts, empowering applications from disaster preparedness to sustainable policy-making.

### 2. What are climate teleconnections and why are they important?
Climate teleconnections are large-scale patterns linking climate events across distant regions. These patterns are vital for forecasting phenomena like El Niño, AMV-driven temperature shifts, or droughts, enabling proactive planning for agriculture, infrastructure, and climate resilience. Using historical and simulated data from the Norwegian Earth System Model (NorESM1-F), the demonstrator identifies these connections, providing actionable insights for long-term climate prediction.

### 3. Which climate indices are analyzed in this demonstrator?
The demonstrator processes 65 climate indices from NorESM1-F simulations, covering:
- **Global Surface Temperature**: `glTSann`, `nhTSann`, `shTSann` for global and hemispheric trends.
- **Sea Surface Temperature (SST)**: `amoSSTann`, `satlSSTann`, `ensoSSTjfm` for oceanic variability.
- **Sea Ice Concentration (SIC)**: `nhSICmar`, `nhSICsep`, `shSICmar`, `shSICsep` for polar ice dynamics.
- **Precipitation**: `nchinaPRjja`, `yrvPRjja`, `ismPRjja` for regional rainfall.
- **Pressure**: `naoPSLjfm`, `eapPSLjfm`, `scpPSLjfm` for atmospheric dynamics.
- **Ocean Circulation**: `AMOCann`, `traBO`, `traBS` for global currents.

See the [Climate Data and Indices](04-understanding-data.md) episode for the complete list.

### 4. What makes this demonstrator different from traditional climate modeling approaches?
Unlike traditional models reliant on linear regression, the NAIC Teleconnection Demonstrator uses advanced ML algorithms—Linear Regression, Random Forests, Multi-Layer Perceptrons (MLPs), and XGBoost—to capture complex, non-linear teleconnections. This approach delivers superior accuracy and scalability for climate research on AMV, PDV, and beyond.

### 5. What are the main goals of the NAIC Teleconnection Demonstrator?
The demonstrator aims to:
- **Uncover Significant Teleconnections**: Identify relationships between indices like AMO and NAO to enhance AMV and PDV understanding.
- **Forecast Climate Indices**: Predict indices decades ahead for robust long-term modeling.
- **Provide Accessible ML Tools**: Enable researchers to apply ML to climate data on cloud VMs.

### 6. How does machine learning help in understanding climate variability?
Machine learning transforms climate variability analysis by:
- **Pattern Recognition**: Detecting subtle patterns in time-series data, such as lagged NAO-AMO relationships.
- **Non-Linear Modeling**: Capturing complex interactions critical for AMV and PDV.
- **Feature Importance**: Highlighting key predictors using models like Random Forests.

The demonstrator predicts indices up to decades ahead using rolling windows, with performance evaluated via correlation coefficients and Mean Absolute Error (MAE), ensuring accurate and interpretable results.

### 7. What datasets are used in the demonstrator?
The demonstrator uses NorESM1-F simulation data, included in the repository:
- **Historical SLOW Forcing (850–2005 AD)**: Low solar variability, volcanic activity, and anthropogenic effects.
- **Historical HIGH Forcing (850–2005 AD)**: High solar variability for comparative analysis.
- **Pre-Industrial Control (PICNTL)**: 1,000 years of constant 1850-level forcing.

These datasets are in the `dataset/` folder, covering 65 indices.

### 8. Can this demonstrator be applied to other regions or indices?
Yes, the demonstrator's modular design supports custom datasets for any region or index. Users can add time-series data to the `dataset/` folder, ensuring the format (time as the first column, followed by indices) matches the NorESM1-F data.

### 9. What are some real-world applications of this analysis?
The demonstrator's insights enable:
- **Decadal Forecasting**: Predict AMV/PDV trends for long-term planning.
- **Agriculture**: Optimize crop strategies using precipitation forecasts.
- **Disaster Preparedness**: Anticipate extreme events like droughts linked to teleconnections.
- **Policy-Making**: Inform climate policies with reliable projections.

### 10. Who can benefit from using this demonstrator?
The demonstrator serves:
- **Climate Scientists**: Exploring AMV/PDV teleconnections.
- **Data Scientists**: Applying ML to climate datasets.
- **Environmental Researchers**: Investigating interdisciplinary impacts.
- **Policy Makers**: Using forecasts for sustainable decisions.
- **Enthusiasts**: Experimenting with ML for climate analysis.

## Category B: Infrastructure & Resources

### 1. What resources do I need for this demonstrator?
The demonstrator is designed to run on NAIC Orchestrator VMs. For most analyses:
- A standard VM with 4-8 CPU cores is sufficient for Linear Regression and Random Forest models
- A GPU-enabled VM is recommended for XGBoost and MLP models for faster training

### 2. Can I run this demonstrator on a local machine?
Yes, for small-scale testing and exploration:
- Install dependencies with `pip install -r requirements.txt`
- Run with reduced parameters (fewer features, shorter lag ranges)

For full analyses with all models and parameter combinations, a VM with more resources is recommended.

### 3. What are the trade-offs between using CPUs and GPUs for ML in climate modeling?
- **GPUs**: Accelerate tensor-heavy models (MLPs, XGBoost) for speed.
- **CPUs**: Preferred for interpretable models (Random Forests, Linear Regression) critical for understanding teleconnections.

The demonstrator supports both, with XGBoost automatically using GPU when available.

## Category C: Prerequisites & Getting Started

### 1. What are the prerequisites to use this demonstrator?
You need:
- **Python 3.8+**: For library compatibility.
- **Basic Python/ML Knowledge**: Familiarity with data handling and models like Random Forests.
- **Jupyter Notebooks**: For interactive analysis via the demonstrator notebook.

### 2. What skills or tools do I need?
- **Git**: For cloning the repository.
- **Python/Jupyter**: For running scripts and notebooks.
- **Command-Line Basics**: For SSH and VM navigation.

### 3. How do I install the required dependencies?
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Where do I get the training data?
NorESM1-F data (SLOW, HIGH, PICNTL) is **included in the repository** in the `dataset/` folder, covering 65 indices. No additional downloads required.

### 5. What is the structure of the project repository?
```
d7.2-Use-case1/
├── dataset/                           # Climate datasets (included)
├── scripts/lrbased_teleconnection/    # ML training scripts
├── results/                           # Output directory with samples
├── demonstrator-v1.orchestrator.ipynb # Interactive notebook
├── utils.py, widgets.py               # Utility modules
└── requirements.txt                   # Python dependencies
```

### 6. How do I run the demonstrator for the first time?
1. Clone the repository:
   ```bash
   git clone https://github.com/NAICNO/wp7-UC1-climate-indices-teleconnection.git
   cd d7.2-Use-case1
   ```

2. Set up environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run a quick test:
   ```bash
   python scripts/lrbased_teleconnection/main.py \
       --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
       --target_feature amo2 \
       --modelname LinearRegression \
       --max_allowed_features 3 \
       --end_lag 30 \
       --n_ensembles 5
   ```

4. Or use Jupyter:
   ```bash
   jupyter lab
   # Open demonstrator-v1.orchestrator.ipynb
   ```

## Category D: Technical Setup & Execution

### 1. How do I set up JupyterLab on the VM?
Use tmux for persistence:
```bash
tmux new -s jupyter
cd ~/d7.2-Use-case1
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
# Detach: Ctrl+B, then D
```

Create SSH tunnel from your local machine:
```bash
ssh -f -N -L 8888:localhost:8888 -i /path/to/key.pem ubuntu@<VM_IP>
```

Then open http://localhost:8888 in your browser.

### 2. Why is my job taking too long?
For long analyses:
- Use tmux to run in background
- Start with fewer features (`--max_allowed_features 3`)
- Use shorter lag ranges (`--end_lag 30`)
- Check `training.log` for progress

### 3. How do I visualize model performance?
Sample visualizations are in `results/`:
- Correlation plots (actual vs. predicted)
- MAE plots for accuracy
- Feature importance plots for key predictors

Use the Jupyter notebook for interactive visualization.

### 4. How do I ensure reproducibility of my results?
- Set random seeds in scripts
- Log configurations with results
- Use Git for version control

Results are consistent with minor stochasticity from model training.

## Category E: Data Access, Sharing & Reproducibility

### 1. How do I get training data for the demonstrator?
All data is **included in this repository**. Simply clone and run:
```bash
git clone https://github.com/NAICNO/wp7-UC1-climate-indices-teleconnection.git
```

### 2. Can I use the data in my own research?
Yes, with attribution to the NAIC Teleconnection Demonstrator and the original NorESM1-F simulation data from NORCE Climate and Environment.

### 3. How do I cite this demonstrator in a publication?
Reference:
> Omrani, N. E., Keenlyside, N., Matthes, K., Boljka, L., Zanchettin, D., Jungclaus, J. H., & Lubis, S. W. (2022). Coupled stratosphere-troposphere-Atlantic multidecadal oscillation and its importance for near-future climate projection. NPJ Climate and Atmospheric Science, 5(1), 59.

### 4. How do I contribute improvements or bug fixes?
Fork the repository, make changes, and submit a merge request via GitLab.

```{keypoints}
- The demonstrator analyzes 65 climate indices using ML
- Four ML models are available: Linear Regression, Random Forest, MLP, XGBoost
- Data comes from NorESM1-F simulations spanning centuries
- All code and data is included - just clone and run
- Results can be used for decadal forecasting, agriculture, and policy-making
```

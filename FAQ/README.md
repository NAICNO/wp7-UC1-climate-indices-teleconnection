# NAIC Teleconnection Demonstrator FAQ

Welcome to the comprehensive FAQ for the NAIC Teleconnection Demonstrator, a Jupyter-based framework that integrates machine learning (ML) to analyze climate teleconnections, with a focus on Atlantic Multidecadal Variability (AMV) and Pacific Decadal Variability (PDV). This document is organized into five categories to guide climate scientists, data scientists, policy makers, and enthusiasts in understanding, setting up, and leveraging this tool for advanced climate research. All information aligns with the project's technical documentation to ensure accuracy and consistency.

## Category A: Understanding the Demonstrator

### 1. What is the NAIC Teleconnection Demonstrator and what problem does it solve?
The NAIC Teleconnection Demonstrator is a platform that combines machine learning to uncover teleconnections -- statistical relationships between climate indices like the Atlantic Multidecadal Oscillation (AMO) and North Atlantic Oscillation (NAO). Traditional climate models often struggle with the non-linear complexity of these relationships, which are critical for predicting long-term climate trends like AMV and PDV. By analyzing 65 climate indices across centuries of data, the demonstrator delivers precise, decadal-scale forecasts, empowering applications from disaster preparedness to sustainable policy-making.

### 2. What are climate teleconnections and why are they important?
Climate teleconnections are large-scale patterns linking climate events across distant regions, such as Pacific sea surface temperatures influencing Northern Hemisphere precipitation. These patterns are vital for forecasting phenomena like El Nino, AMV-driven temperature shifts, or droughts, enabling proactive planning for agriculture, infrastructure, and climate resilience. Using historical and simulated data from the Norwegian Earth System Model (NorESM1-F), the demonstrator identifies these connections, providing actionable insights for long-term climate prediction.

### 3. Which climate indices are analyzed in this demonstrator?
The demonstrator processes 65 climate indices from NorESM1-F simulations, covering:
- **Global Surface Temperature**: `glTSann`, `nhTSann`, `shTSann` for global and hemispheric trends.
- **Sea Surface Temperature (SST)**: `amoSSTann`, `satlSSTann`, `ensoSSTjfm` for oceanic variability.
- **Sea Ice Concentration (SIC)**: `nhSICmar`, `nhSICsep`, `shSICmar`, `shSICsep` for polar ice dynamics.
- **Precipitation**: `nchinaPRjja`, `yrvPRjja`, `ismPRjja` for regional rainfall.
- **Pressure**: `naoPSLjfm`, `eapPSLjfm`, `scpPSLjfm` for atmospheric dynamics.
- **Ocean Circulation**: `AMOCann`, `traBO`, `traBS` for global currents.
A detailed list is available in the project's [CLIMATE-INDICES.md](documentations/CLIMATE-INDICES.md) file, supporting comprehensive AMV and PDV analysis.

### 4. What makes this demonstrator different from traditional climate modeling approaches?
Unlike traditional models reliant on linear regression, the NAIC Teleconnection Demonstrator uses advanced ML algorithms -- Linear Regression, Random Forests, Multi-Layer Perceptrons (MLPs), and XGBoost -- to capture complex, non-linear teleconnections. It efficiently analyzes vast datasets, reducing computation time for millions of index combinations. This approach delivers superior accuracy and scalability, advancing climate research for AMV, PDV, and beyond.

### 5. What are the main goals of the NAIC Teleconnection Demonstrator?
The demonstrator aims to:
- **Uncover Significant Teleconnections**: Identify relationships between indices like AMO and NAO to enhance AMV and PDV understanding.
- **Forecast Climate Indices**: Predict indices decades ahead for robust long-term modeling.
- **Enable Scalable Computation**: Leverage NAIC Orchestrator VMs to accelerate computations, enabling scalable analysis of complex climate data.
These goals position the demonstrator as a valuable tool for climate prediction.

### 6. How does machine learning help in understanding climate variability?
Machine learning transforms climate variability analysis by:
- **Pattern Recognition**: Detecting subtle patterns in time-series data, such as lagged NAO-AMO relationships.
- **Non-Linear Modeling**: Capturing complex interactions critical for AMV and PDV.
- **Feature Importance**: Highlighting key predictors using models like Random Forests.
The demonstrator predicts indices up to decades ahead using rolling windows, with performance evaluated via correlation coefficients and Mean Absolute Error (MAE), ensuring accurate and interpretable results.

### 7. What datasets are used in the demonstrator?
The demonstrator uses NorESM1-F simulation data, including:
- **Historical SLOW Forcing (850-2005 AD)**: Low solar variability, volcanic activity, and anthropogenic effects.
- **Historical HIGH Forcing (850-2005 AD)**: High solar variability for comparative analysis.
- **Pre-Industrial Control (PICNTL)**: 1,000 years of constant 1850-level forcing.
Stored in the `dataset/` folder, these datasets cover 65 indices, providing a robust foundation for teleconnection analysis.

### 8. Can this demonstrator be applied to other regions or indices?
Yes, the demonstrator's modular design supports custom datasets for any region or index. Users can add time-series data to the `dataset/` folder, ensuring the format (time as the first column, followed by indices) matches the NorESM1-F data. Minor script tweaks may be needed, enabling global applications beyond AMV and PDV, such as regional weather or novel indices.

### 9. What are some real-world applications of this analysis?
The demonstrator's insights enable:
- **Decadal Forecasting**: Predict AMV/PDV trends for long-term planning.
- **Agriculture**: Optimize crop strategies using precipitation forecasts.
- **Disaster Preparedness**: Anticipate extreme events like droughts linked to teleconnections.
- **Policy-Making**: Inform climate policies with reliable projections.
Sample results in the `results/` folder, like `amoSSTann` predictions, highlight validated teleconnections for practical impact.

### 10. Who can benefit from using this demonstrator?
The demonstrator serves:
- **Climate Scientists**: Exploring AMV/PDV teleconnections.
- **Data Scientists**: Applying ML to climate datasets.
- **Environmental Researchers**: Investigating interdisciplinary impacts.
- **Policy Makers**: Using forecasts for sustainable decisions.
- **Enthusiasts**: Experimenting with ML-based climate analysis.
Its accessible design empowers diverse users to advance climate research.

## Category B: Infrastructure & Resources

### 1. How do I know what kind of resources I need for a demonstrator like this (parallel, CPU, GPU)?
Analyzing 65 climate indices with over 1,300 data points each generates millions of combinations. Parallel CPU processing is the primary approach, as the computational bottleneck is task volume, not data size. While GPUs can accelerate MLPs or XGBoost, CPUs are prioritized for interpretable models like Random Forests, ensuring robust teleconnection analysis on NAIC Orchestrator VMs.

### 2. Where can I find compute resources to run this?
The NAIC Teleconnection Demonstrator runs on NAIC Orchestrator VMs. Provision a VM at [https://orchestrator.naic.no](https://orchestrator.naic.no) with optional GPU support. The Orchestrator portal provides details on available VM configurations including CPU/GPU options and access procedures.

### 3. How much does it cost to run this?
NAIC Orchestrator VMs are provisioned through the NAIC portal. Check the [NAIC Orchestrator](https://orchestrator.naic.no) for current resource availability and allocation policies.

### 4. What are the trade-offs between using CPUs and GPUs for ML in climate modeling?
- **GPUs**: Accelerate tensor-heavy models (MLPs, XGBoost) for speed.
- **CPUs**: Preferred for interpretable models (Random Forests, smaller MLPs) critical for understanding teleconnections.
The demonstrator prioritizes CPUs for transparency, ensuring results are scientifically meaningful. GPUs are optional and used automatically when available.

### 5. How do I request a VM with GPU support?
Provision a VM with GPU support through the [NAIC Orchestrator portal](https://orchestrator.naic.no). Select a GPU-enabled flavor when creating your VM instance. CPUs are sufficient for most tasks, aligning with the demonstrator's design for accessibility.

### 6. Can I run this demonstrator on a local machine or cloud instead?
- **Local Machines**: Suitable for small-scale testing but may be slow for full analyses with many index combinations.
- **Cloud Platforms**: Feasible but may be costly for large-scale runs.
- **NAIC Orchestrator VMs** (recommended): Optimized for this workload with easy provisioning and optional GPU support.

### 7. Can I run long-running experiments without interruption?
Yes. Use `tmux` sessions on NAIC Orchestrator VMs to run training jobs in the background. Detach from the session and reconnect later to check progress. See the main README for tmux usage examples.

## Category C: Prerequisites & Getting Started

### 1. What are the prerequisites to adopt this demonstrator for my work?
You need:
- **Python 3.8+**: For library compatibility.
- **Basic Python/ML Knowledge**: Familiarity with data handling and models like Random Forests.
- **Jupyter Notebooks**: For interactive analysis via the demonstrator notebook.
- **SSH Access**: To connect to an NAIC Orchestrator VM.
These requirements ensure accessibility for diverse users.

### 2. What skills or tools do I need to start using this demonstrator?
- **Git**: For cloning (`git@github.com:NAICNO/wp7-UC1-climate-indices-teleconnection.git`) and version control.
- **Python/Jupyter**: For running scripts and notebooks.
- **Command-Line Basics**: For SSH access and navigation.
Clear documentation ensures a smooth start.

### 3. How do I install the required dependencies?
Per the project's installation steps:
1. Clone the repository.
2. Run the setup script: `./setup.sh`.
3. Activate the environment: `source venv/bin/activate`.
This installs `numpy`, `pandas`, `scikit-learn`, `xgboost`, etc.

### 4. What datasets are required and where do I get them?
NorESM1-F data (SLOW, HIGH, PICNTL simulations) is included in the `dataset/` folder. Add custom datasets in the same format (time as first column, followed by indices) for flexibility.

### 5. What is the structure of the project repository?
- **dataset/**: NorESM1-F data and configurations.
- **scripts/**: Python ML training scripts.
- **results/**: CSV files and visualizations.
- **notebooks/**: Jupyter notebooks, with `demonstrator-v1.orchestrator.ipynb` as the entry point.

### 6. How do I run the demonstrator for the first time?
1. Clone: `git clone git@github.com:NAICNO/wp7-UC1-climate-indices-teleconnection.git`.
2. Run `./setup.sh` and `source venv/bin/activate`.
3. Open `demonstrator-v1.orchestrator.ipynb` in Jupyter.
4. Or run from the command line:
```bash
python scripts/lrbased_teleconnection/main.py --target_feature=amo3 --modelname=LinearRegression --splitsize=0.6 --step_lag=5 --data_file=dataset/noresm-f-p1000_shigh_new_jfm.csv --max_allowed_features=10 --with_mean_feature
```

### 7. What are common setup issues and how do I fix them?
- **Python Version**: Use Python 3.8 or newer (3.10+ recommended).
- **Dependencies**: Re-run `pip install -r requirements.txt`.
- **GPU Not Detected**: CPU-only mode is used automatically; this is fine for most models.
- **SSH Connection Issues**: Verify your VM is running and SSH key permissions are correct (`chmod 600 <key>`).

## Category D: Technical Setup & Execution

### 1. How do I set up JupyterLab on an NAIC Orchestrator VM?
Start Jupyter Lab on the VM and use SSH port forwarding to access it from your local browser:
```bash
# On the VM
jupyter lab --no-browser --ip=127.0.0.1 --port=8888

# On your local machine
ssh -N -L 8888:localhost:8888 -i <SSH_KEY_PATH> ubuntu@<VM_IP>
```
Then open `http://localhost:8888` in your browser.

### 2. How do I use GPUs to perform tasks in this demonstrator?
Provision a GPU-enabled VM from the NAIC Orchestrator portal. The demonstrator auto-detects GPUs for XGBoost/MLPs, but CPUs are prioritized for interpretability.

### 3. What does the error "library not found" mean and how do I fix it?
Causes:
- Missing `requirements.txt` libraries.
- Version mismatches.
Fixes:
- Reinstall: `pip install -r requirements.txt`.
- Ensure you activated the virtual environment: `source venv/bin/activate`.

### 4. Why is my job taking too long or not completing?
Check `results/` CSV logs for progress. Consider reducing the number of ensemble runs (`--n_ensembles`), limiting features (`--max_allowed_features`), or reducing the lag range (`--end_lag`).

### 5. What does "Parallel Evaluation" mean in this context?
Each model-index pair is processed independently, enabling parallel analysis across multiple runs for scalability.

### 6. How do I modify the lag or feature settings in the model?
Use:
```bash
python scripts/lrbased_teleconnection/main.py --step_lag=5
```
Adjusts lagged data for predictions (e.g., 5 years).

### 7. How do I visualize model performance?
Visualizations in `results/` include:
- Correlation plots (actual vs. predicted).
- MAE plots for accuracy.
- Feature importance plots for key predictors.

### 8. Can I integrate new ML models into the pipeline?
Yes, extend `scripts/lrbased_teleconnection/main.py` with scikit-learn/XGBoost models or custom preprocessing for flexibility.

### 9. How do I ensure reproducibility of my results?
- Set random seeds.
- Log configurations in `results/`.
- Use Git for version control.
Results are highly consistent despite minor stochasticity.

## Category E: Data Access, Sharing & Reproducibility

### 1. How do I get training data for the demonstrator?
NorESM1-F data (SLOW, HIGH, PICNTL) is in the `dataset/` folder, covering 65 indices.

### 2. Where do I upload my results?
Upload `results/` CSVs to Google Drive, institutional storage, or repositories like Zenodo.

### 3. How do I share my results with others?
Share the demonstrator notebook with embedded descriptions, visualizations, and context for comprehensive communication.

### 4. Can I use the benchmarking data in my own research?
Yes, with attribution to the NAIC Teleconnection Demonstrator, providing valuable teleconnection insights.

### 5. How do I cite this demonstrator in a publication?
Cite as "NAIC Teleconnection Demonstrator" (DOI pending). Contact the NAIC team for guidance.

### 6. Can the demonstrator authors be co-authors on my paper?
Yes, if they contribute significantly. Contact the NAIC team to collaborate.

### 7. How do I anonymize data before sharing it?
Remove any VM-specific references or rename datasets as needed. The climate data is open and non-personal.

### 8. What's the best way to collaborate on extending the demonstrator?
Use Git: create branches, open issues, submit pull requests for trackable collaboration.

### 9. How do I contribute improvements or bug fixes?
Fork the repository, make changes, and submit a pull request for NAIC team review.

### 10. What's the roadmap for future updates to the demonstrator?
Future updates depend on community interest, potentially adding models or datasets to enhance AMV/PDV analysis.

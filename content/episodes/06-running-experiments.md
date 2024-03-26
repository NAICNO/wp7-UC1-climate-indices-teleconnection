# Running Experiments

```{objectives}
- Run the main analysis script
- Use background training for long runs
- Understand the command line options
```

## Usage

### 1. Activate the Environment

```bash
cd ~/wp7-UC1-climate-indices-teleconnection
source venv/bin/activate
```

### 2. Run the Main Analysis Script

```bash
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname LinearRegression \
    --max_allowed_features 6 \
    --end_lag 50
```

### 3. Interactive Exploration with Jupyter

```bash
jupyter lab
# Open demonstrator-v1.orchestrator.ipynb
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--data_file` | Path to dataset CSV | `dataset/noresm-f-p1000_slow_new_jfm.csv` |
| `--target_feature` | Variable to predict | `amo2`, `amo3`, `AMOCann` |
| `--modelname` | ML model to use | `LinearRegression`, `RandomForestRegressor`, `MLPRegressor`, `XGBRegressor` |
| `--max_allowed_features` | Max features for model | `6`, `10` |
| `--end_lag` | Maximum lag in years | `50`, `100` |
| `--step_lag` | Lag step size | `5` |
| `--splitsize` | Train/test split ratio | `0.6` |
| `--n_ensembles` | Number of ensemble runs | `10`, `100` |
| `--with_mean_feature` | Include mean feature | Flag (no value) |

## Background Training (Long Runs)

For long-running experiments, use tmux sessions:

```bash
# Start background training
tmux new-session -d -s training 'cd ~/wp7-UC1-climate-indices-teleconnection && source venv/bin/activate && \
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname LRforcedPSO \
    --max_allowed_features 6 \
    --end_lag 100 \
    --n_ensembles 100 2>&1 | tee training.log'

# Monitor progress
tail -f training.log

# Attach to session
tmux attach -t training

# Detach: Ctrl+B, then D
```

## Example Experiments

### Quick Test (Linear Regression)

```bash
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname LinearRegression \
    --max_allowed_features 3 \
    --end_lag 30 \
    --n_ensembles 5
```

### Full Analysis (Multiple Models)

```bash
# Linear Regression
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname LinearRegression \
    --max_allowed_features 6 \
    --end_lag 100

# Random Forest
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname RandomForestRegressor \
    --max_allowed_features 6 \
    --end_lag 100

# XGBoost (GPU accelerated)
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname XGBRegressor \
    --max_allowed_features 6 \
    --end_lag 100
```

## Result File Format

Results are saved to the `results/` directory:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `target_feature` | Predicted variable |
| `max_lag` | Maximum lag in years |
| `corr_score` | Correlation coefficient |
| `mae_score` | Mean Absolute Error |
| `selected_features` | Features used by model |

```{keypoints}
- Run experiments using main.py with appropriate parameters
- Use tmux for long-running background experiments
- Results are saved to the results/ directory
- Use Jupyter notebook for interactive exploration
```

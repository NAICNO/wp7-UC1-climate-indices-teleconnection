# AI Agent Instructions for Teleconnections Demonstrator

This file provides structured instructions for AI coding assistants (Claude, Copilot, etc.) to set up and run the teleconnections demonstrator.

> **Also available:** `AGENT.yaml` - Same instructions in YAML format for programmatic parsing.

## Prerequisites

- SSH access to NAIC Orchestrator VM
- VM IP address and SSH key path

## Step 1: Connect to VM and Check Environment

```bash
# SSH to VM
ssh -i <SSH_KEY_PATH> ubuntu@<VM_IP>

# Check if module system is available (NAIC VMs)
module avail python/3 2>/dev/null || echo "No module system"

# Check GPU availability
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"

# Check Python version
python3 --version
```

## Step 2: Initialize VM (if needed)

**Option A: If module system is available (recommended):**
```bash
# Load Python module
module load Python/3.11.5-GCCcore-13.2.0

# Verify
python3 --version
```

**Option B: If no module system (install packages):**
```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev
```

## Step 3: Clone and Setup

```bash
# Clone repository
git clone https://github.com/NAICNO/wp7-UC1-climate-indices-teleconnection.git

# Enter directory
cd d7.2-Use-case1

# Run setup script (auto-detects module system, checks GPU, creates venv)
./setup.sh

# Activate environment
source venv/bin/activate
```

## Step 4: Verify GPU (if available)

```bash
# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Step 5: Run Jupyter Notebook

```bash
# Start Jupyter Lab (on VM)
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
```

Then create SSH tunnel from local machine:
```bash
# Verbose mode (shows connection status)
ssh -v -N -L 8888:localhost:8888 -i <SSH_KEY_PATH> ubuntu@<VM_IP>

# If port 8888 is in use, use alternative port:
ssh -v -N -L 9999:localhost:8888 -i <SSH_KEY_PATH> ubuntu@<VM_IP>
```

> **Note:** The tunnel will appear to "hang" after connecting - this is normal! The tunnel is active. Keep the terminal open while using Jupyter. Press `Ctrl+C` to close.

Open in browser: `http://localhost:8888/lab/tree/demonstrator-v1.orchestrator.ipynb`

## Step 6: Run Command Line Test

```bash
python scripts/lrbased_teleconnection/main.py \
    --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
    --target_feature amo2 \
    --modelname LinearRegression \
    --max_allowed_features 3 \
    --end_lag 30 \
    --n_ensembles 5
```

## Expected Output

- Results saved to `results/` directory
- CSV file with columns: model, target_feature, max_lag, corr_score, mae_score, selected_features
- Correlation score > 0.7 indicates good model performance

## Directory Structure

```
d7.2-Use-case1/
├── vm-init.sh                  # VM environment check/setup
├── setup.sh                    # Python environment setup (auto-detects modules)
├── venv/                       # Created by setup.sh
├── dataset/                    # Input data (included)
│   ├── noresm-f-p1000_slow_new_jfm.csv
│   ├── noresm-f-p1000_shigh_new_jfm.csv
│   └── noresm-f-p1000_picntrl_new_jfm.csv
├── scripts/lrbased_teleconnection/
│   └── main.py                 # Main training script
├── results/                    # Output directory
└── demonstrator-v1.orchestrator.ipynb  # Interactive notebook
```

## Available Parameters

| Parameter | Required | Default | Values |
|-----------|----------|---------|--------|
| --data_file | Yes | - | Path to CSV in dataset/ |
| --target_feature | Yes | - | amo1, amo2, amo3, AMOCann, etc. |
| --modelname | Yes | - | LinearRegression, RandomForestRegressor, MLPRegressor, XGBRegressor |
| --max_allowed_features | No | 6 | 3-32 |
| --end_lag | No | 50 | 20-150 |
| --n_ensembles | No | 10 | 1-200 |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `git: command not found` | `sudo apt install -y git` or check module system |
| `python3: command not found` | `module load Python/3.11.5-GCCcore-13.2.0` |
| Python version too old | Load newer Python module |
| No GPU detected | CPU-only mode will be used automatically |
| CUDA errors | Run `./vm-init.sh` to setup CUDA symlinks |

## Verification

After running, check:
1. `ls results/` - should contain new CSV and PNG files
2. CSV should have corr_score and mae_score columns
3. No Python errors in output

# Setting Up the Environment

```{objectives}
- Connect to your VM via SSH
- Initialize a fresh VM with required packages
- Clone the repository and set up the Python environment
- Start Jupyter Lab with SSH tunneling
```

## 1. Connect to Your VM

Connect to your VM using SSH (see Episode 02 for Windows-specific instructions):

````{tabs}
```{tab} macOS / Linux / Git Bash
chmod 600 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

```{note}
**Windows users**: If you see "Permissions for key are too open", fix the key permissions first. See Episode 02, Step 7 for detailed instructions. Git Bash is recommended — it supports `chmod` natively.
```

## 2. Initialize Fresh VM (Run Once)

On a fresh NAIC VM, install required system packages:

```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev
```

This installs:
- `git` -- For cloning the repository
- `build-essential` -- Compiler toolchain (gcc, make)
- `python3-dev`, `python3-venv`, `python3-pip` -- Python development tools
- `libssl-dev`, `zlib1g-dev` -- Required for building Python packages

## 3. Clone and Setup

```bash
git clone https://github.com/NAICNO/wp7-UC1-climate-indices-teleconnection.git
cd wp7-UC1-climate-indices-teleconnection
./setup.sh
source venv/bin/activate
```

The `setup.sh` script automatically:
- Creates a Python virtual environment
- Installs all dependencies from `requirements.txt`

## 4. Start Jupyter Lab (Optional)

For interactive exploration, start Jupyter Lab:

```bash
# Use tmux for persistence
tmux new -s jupyter
cd ~/wp7-UC1-climate-indices-teleconnection
source venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 
# Detach with Ctrl+B, then D
```

## 5. Create SSH Tunnel (on your local machine)

To access Jupyter Lab from your local browser, create an SSH tunnel. Open a **new terminal** on your local machine (not the VM):

````{tabs}
```{tab} macOS / Linux / Git Bash
# Verbose mode (recommended - shows connection status)
ssh -v -N -L 8888:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -v -N -L 8888:localhost:8888 -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

> **Note:** The tunnel will appear to "hang" after connecting -- this is normal! It means the tunnel is active. Keep the terminal open while using Jupyter.

**If port 8888 is already in use**, use an alternative port:

```bash
ssh -v -N -L 9999:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
# Then access via http://localhost:9999
```

Then navigate to: **http://localhost:8888/lab/tree/demonstrator-v1.orchestrator.ipynb**

To close the tunnel, press `Ctrl+C` in the terminal.

## Project Structure

After cloning, you will have:

```
wp7-UC1-climate-indices-teleconnection/
├── vm-init.sh                         # VM initialization (optional)
├── setup.sh                           # Python environment setup
├── dataset/                           # Climate datasets (included)
│   ├── noresm-f-p1000_slow_new_jfm.csv
│   ├── noresm-f-p1000_shigh_new_jfm.csv
│   └── noresm-f-p1000_picntrl_new_jfm.csv
├── scripts/lrbased_teleconnection/    # ML training scripts
│   ├── main.py
│   ├── models.py
│   ├── dataloader.py
│   ├── libs.py
│   ├── evaluation.py
│   └── plotting.py
├── results/                           # Output directory with samples
├── demonstrator-v1.orchestrator.ipynb # Interactive notebook
├── utils.py
├── widgets.py
└── requirements.txt
```

## Dependencies

The following packages are installed via `setup.sh`:

- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting with GPU support
- `scipy` - Scientific computing
- `tqdm` - Progress bars
- `matplotlib`, `seaborn` - Visualization
- `pycwt` - Wavelet analysis
- `jupyterlab`, `ipywidgets` - Interactive notebooks

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `git: command not found` | Run VM init: `sudo apt install -y git` |
| Connection refused | Verify VM is running with `ping <VM_IP>` |
| Permission denied | `chmod 600 /path/to/your-key.pem` |
| SSH "Permissions too open" (Windows) | Use Git Bash (`chmod 600`) or fix via icacls — see Episode 02 |
| SSH connection timed out | Your IP may not be whitelisted — add it at orchestrator.naic.no |
| Host key error | `ssh-keygen -R <VM_IP>` (VM IP changed) |
| Jupyter not accessible | Check tunnel is running; verify correct port |
| Port 8888 already in use | Use alternative port: `-L 9999:localhost:8888` |
| SSH tunnel appears to hang | This is normal -- tunnel is active, keep terminal open |
| Import errors | Verify venv is activated: `which python` |

```{keypoints}
- Set SSH key permissions with `chmod 600` before connecting (use Git Bash on Windows)
- Initialize fresh VMs with `sudo apt install -y build-essential git python3-dev python3-venv`
- Clone this repository directly -- all code and data are included
- Run `./setup.sh` to automatically set up the Python environment
- Use tmux for persistent Jupyter Lab sessions
- Create an SSH tunnel to access Jupyter from your local browser
- Windows users: Git Bash is recommended for the best experience with SSH and Unix commands
```

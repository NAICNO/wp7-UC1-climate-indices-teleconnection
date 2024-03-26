Downloads & Quick Reference
===========================

.. important::
   This repository is self-contained. After cloning, run ``./setup.sh`` to set up the environment automatically.

Setup Script
------------

Run after cloning to set up the environment:

.. code-block:: bash

   git clone https://github.com/NAICNO/wp7-UC1-climate-indices-teleconnection.git
   cd wp7-UC1-climate-indices-teleconnection
   ./setup.sh

Quick Reference Card
--------------------

**Available Models**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model Name
     - Description
   * - ``LinearRegression``
     - Simple baseline for linear relationships
   * - ``RandomForestRegressor``
     - Ensemble model with decision trees
   * - ``MLPRegressor``
     - Neural network for non-linear patterns
   * - ``XGBRegressor``
     - Gradient boosting with GPU support

**Command Line Options**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Option
     - Default
     - Description
   * - ``--data_file``
     - Required
     - Path to dataset CSV
   * - ``--target_feature``
     - Required
     - Variable to predict (e.g., amo2)
   * - ``--modelname``
     - Required
     - ML model to use
   * - ``--max_allowed_features``
     - 6
     - Maximum features for model
   * - ``--end_lag``
     - 50
     - Maximum lag in years
   * - ``--step_lag``
     - 5
     - Lag step size
   * - ``--splitsize``
     - 0.6
     - Train/test split ratio
   * - ``--n_ensembles``
     - 10
     - Number of ensemble runs
   * - ``--with_mean_feature``
     - False
     - Include mean feature (flag)

**Available Datasets**

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Dataset
     - Description
   * - ``dataset/noresm-f-p1000_slow_new_jfm.csv``
     - Historical SLOW forcing (850-2005 AD)
   * - ``dataset/noresm-f-p1000_shigh_new_jfm.csv``
     - Historical HIGH forcing (850-2005 AD)
   * - ``dataset/noresm-f-p1000_picntrl_new_jfm.csv``
     - Pre-industrial control (1000 years)

**Common Target Features**

``amo1``, ``amo2``, ``amo3``, ``AMOCann``, ``amoSSTann``, ``naoPSLjfm``, ``ensoSSTjfm``

Example Commands
----------------

**Quick Test (< 1 minute)**

.. code-block:: bash

   python scripts/lrbased_teleconnection/main.py \
       --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
       --target_feature amo2 \
       --modelname LinearRegression \
       --max_allowed_features 3 \
       --end_lag 30 \
       --n_ensembles 5

**Full Analysis with Random Forest**

.. code-block:: bash

   python scripts/lrbased_teleconnection/main.py \
       --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
       --target_feature amo2 \
       --modelname RandomForestRegressor \
       --max_allowed_features 6 \
       --end_lag 100

**Background Training (tmux)**

.. code-block:: bash

   tmux new-session -d -s training 'source venv/bin/activate && \
   python scripts/lrbased_teleconnection/main.py \
       --data_file dataset/noresm-f-p1000_slow_new_jfm.csv \
       --target_feature amo2 \
       --modelname XGBRegressor \
       --max_allowed_features 6 \
       --end_lag 100 2>&1 | tee training.log'

   # Monitor: tail -f training.log
   # Attach: tmux attach -t training

For AI Coding Assistants
------------------------

If you're using an AI coding assistant (Claude Code, GitHub Copilot, Cursor, etc.), the repository includes machine-readable instruction files:

- ``AGENT.md`` - Markdown format (human and agent readable)
- ``AGENT.yaml`` - YAML format (structured data for programmatic parsing)

These files contain step-by-step instructions that agents can follow to:

1. Set up the environment on the VM
2. Run the Jupyter notebook
3. Execute command-line experiments
4. Verify results

**Quick prompt for your AI assistant:**

.. code-block:: text

   Read AGENT.md and help me run the teleconnections demonstrator on my NAIC VM.
   VM IP: <your_vm_ip>
   SSH Key: <path_to_your_key.pem>

The agent will execute the setup and run experiments based on the structured instructions.

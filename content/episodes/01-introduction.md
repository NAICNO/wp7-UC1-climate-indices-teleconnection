# Introduction to Climate Teleconnections

```{objectives}
- Understand what the NAIC Teleconnection Demonstrator does
- Learn about teleconnections and why ML is useful
- Know the project objectives
```

## Overview

This repository provides a complete framework for analyzing climate indices using advanced machine learning (ML) models. It includes scripts for data preparation, model training, and results analysis. The goal is to identify teleconnections between climate indices that can improve climate predictions and long-term forecasts.

Teleconnections are large-scale patterns of climate variability, and understanding them is critical for predicting long-term climate changes. The focus here is on the Atlantic Multidecadal Variability (AMV) and Pacific Decadal Variability (PDV) in the Northern Hemisphere, but the methods can be applied globally.

The framework leverages ML to identify relationships between climate indices (known as "predictor variables") and uses these to forecast future climate conditions (the "target variable"). This is done through both linear and non-linear models, and model performance is assessed with correlation coefficients and Mean Absolute Error (MAE).

### Key File

The `demonstrator-v1.orchestrator.ipynb` notebook showcases this process, focusing on how climate indices like the Atlantic Multidecadal Oscillation (AMO) are predicted from other climate variables such as the North Atlantic Oscillation (NAO), sea ice concentration, and ocean circulation.

## Objectives

The primary goals of this project are to:

- **Identify significant teleconnections:** Using ML models, the project aims to discover relationships between climate indices that can help predict future climate trends.
- **Predict climate indices for future periods:** Forecast climate indices several decades into the future to improve long-term climate modeling.
- **Provide accessible ML tools:** Enable researchers to apply ML to climate data on NAIC Orchestrator VMs.

## Self-Contained Repository

This repository is self-contained. Everything you need is included:

| Component | Location |
|-----------|----------|
| Climate datasets | `dataset/` (3 NorESM1-F simulations) |
| ML training scripts | `scripts/lrbased_teleconnection/` |
| Interactive notebook | `demonstrator-v1.orchestrator.ipynb` |
| Sample results | `results/` |
| Dependencies | `requirements.txt` |

Simply clone the repository and follow the setup instructions to get started.

## Using AI Coding Assistants

If you're using an AI coding assistant like **Claude Code**, **GitHub Copilot**, or **Cursor**, the repository includes an `AGENT.md` file with machine-readable instructions. Simply tell your assistant:

> "Read AGENT.md and help me run the teleconnections demonstrator on my NAIC VM."

The agent will be able to set up the environment and run experiments automatically. See the [Quick Reference](../downloads/index.rst) page for more details.

## What You Will Learn

| Episode | Topic |
|---------|-------|
| 02 | Provisioning a NAIC VM |
| 03 | Setting up the environment |
| 04 | Climate data and indices |
| 05 | ML methodology |
| 06 | Running experiments |
| 07 | Analyzing results |
| 08 | FAQ |

## Resources

- NAIC Portal: https://orchestrator.naic.no
- VM Workflows Guide: https://training.pages.sigma2.no/tutorials/naic-cloud-vm-workflows/
- This Repository: https://github.com/NAICNO/wp7-UC1-climate-indices-teleconnection

```{keypoints}
- Teleconnections are large-scale patterns of climate variability
- Focus is on Atlantic Multidecadal Variability (AMV) and Pacific Decadal Variability (PDV)
- ML models identify relationships between climate indices to forecast future conditions
- Performance is assessed with correlation coefficients and Mean Absolute Error (MAE)
- All code and data are included in this repository
```

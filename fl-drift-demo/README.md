# Federated LLM Drift Detection and Recovery System

A reproducible pipeline for detecting and recovering from data drift in federated Large Language Model (LLM) deployments across distributed nodes.

## Overview

This system addresses the challenge where input data from distributed clients gradually diverges due to evolving local patterns, resulting in degraded accuracy and inconsistent outputs. The solution implements multi-level drift detection and automated recovery mechanisms.

## Features

- **Multi-level Drift Detection**: Local client-level and global server-level monitoring
- **Automated Recovery**: Adaptive aggregation strategies (FedAvg → FedTrimmedAvg)
- **Synthetic Drift Injection**: For testing and validation
- **Comprehensive Metrics**: Accuracy, fairness gap, detection delay, recovery rate

## Quick Start

cd fl-drift-demo
pip install -r requirements.txt

1. Activate virtual environment:
   ```bash
   source fl_env/bin/activate
   ```

2. Run the simulation:
   ```bash
   flwr run
   ```

## Project Structure

```
fl-drift-demo/
├── pyproject.toml          # Project configuration
├── fed_drift/              # Main package
│   ├── client.py           # Drift-aware client implementation
│   ├── server.py           # Drift-aware server strategy
│   ├── data.py             # Data preparation and drift injection
│   ├── models.py           # Model definitions
│   └── drift_detection.py  # Drift detection algorithms
├── tests/                  # Unit tests
├── results/                # Simulation results
└── README.md              # This file
```

## Configuration

Key parameters in `pyproject.toml`:
- `num_supernodes = 10`: Number of federated clients
- `num_rounds = 50`: Total training rounds
- Drift injection at round 25

## Dependencies

- Flower (flwr): Federated learning framework
- Transformers + PyTorch: LLM implementation
- Alibi Detect: Server-side drift detection
- Evidently + River: Client-side drift detection
- nlpaug: Synthetic drift injection# devops-prj

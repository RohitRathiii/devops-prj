# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Federated LLM Drift Detection and Recovery System** - a comprehensive Python package that implements multi-level drift detection and automated mitigation strategies for federated learning deployments with Large Language Models. The system uses the Flower framework for federated learning orchestration and implements sophisticated drift detection mechanisms at both client and server levels.

## Development Environment Setup

### Virtual Environment
The project uses a Python virtual environment named `fl_env`:
```bash
source fl_env/bin/activate  # Activate environment
deactivate                  # Deactivate when done
```

### Core Dependencies
- **Flower Framework**: `flwr[simulation]` for federated learning simulation
- **ML Stack**: `transformers[torch]`, `torch`, `datasets` for BERT-tiny model training
- **Drift Detection**: `alibi-detect` (MMD), `evidently` (data drift), `river` (ADWIN)
- **Data Augmentation**: `nlpaug` for synthetic drift injection
- **Testing**: `pytest` for unit tests

### Common Commands

**Run the full simulation:**
```bash
python main.py --rounds 50 --clients 10
```

**Validate system setup:**
```bash
python validate_setup.py
```

**Run with custom configuration:**
```bash
python main.py --config custom_config.yaml
```

**Run tests:**
```bash
pytest tests/ -v
python main.py --mode test
```

**Quick validation mode:**
```bash
python main.py --mode validate
```

**Run with Flower simulation:**
```bash
flwr run
```

## Project Architecture

### Package Structure
```
fl-drift-demo/
├── fed_drift/              # Main package - core drift detection system
│   ├── client.py           # DriftDetectionClient - federated client with drift monitoring
│   ├── server.py           # DriftAwareFedAvg - server strategy with mitigation
│   ├── data.py             # FederatedDataLoader, DriftInjector - data handling
│   ├── models.py           # BERTClassifier - BERT-tiny wrapper
│   ├── drift_detection.py  # Multi-level detectors (ADWIN, MMD, Evidently)
│   ├── simulation.py       # FederatedDriftSimulation - end-to-end orchestration
│   └── config.py           # ConfigManager - YAML/JSON configuration
├── tests/                  # Unit test suite
├── main.py                 # CLI entry point with argparse
└── validate_setup.py       # System validation script
```

### Key Components

**Multi-Level Drift Detection Architecture:**
- **Client-Side**: ADWIN concept drift detection + Evidently data drift analysis
- **Server-Side**: MMD drift test on aggregated embeddings
- **Integration**: Drift signals embedded in standard FL evaluation messages

**Adaptive Mitigation Strategy:**
- **Baseline**: FedAvg for normal operation
- **Mitigation**: FedTrimmedAvg with β=0.2 trimming after drift detection
- **Triggers**: Global MMD p-value < 0.05 OR >30% clients report drift

**Synthetic Drift Injection:**
- **Vocabulary Shift**: nlpaug synonym replacement (30% intensity)
- **Concept Drift**: Label noise injection (20% noise rate)
- **Distribution Shift**: Class imbalance through selective sampling

### Data Flow
1. **Training Phase**: Clients train BERT-tiny on AG News dataset partitions
2. **Monitoring Phase**: Continuous drift detection during federated rounds
3. **Detection Phase**: Multi-level analysis triggers mitigation decision
4. **Recovery Phase**: Switch to robust aggregation strategy
5. **Validation Phase**: Monitor recovery effectiveness

## Configuration Management

### Config File Structure
The system uses YAML/JSON configuration with these key sections:
- `model`: BERT-tiny settings, training hyperparameters
- `federated`: Client count, partitioning strategy (Dirichlet α=0.5)
- `drift`: Injection round, intensity, affected clients, drift types
- `simulation`: Flower simulation parameters, resource allocation
- `drift_detection`: ADWIN delta, MMD p-value thresholds

### CLI Override Support
Main parameters can be overridden via command line:
- `--rounds N`: Override training rounds
- `--clients N`: Override client count
- `--drift-round N`: Override drift injection round
- `--config FILE`: Use custom configuration file

## Testing Strategy

### Test Categories
- **Unit Tests**: Individual component validation (detectors, strategies)
- **Integration Tests**: End-to-end pipeline execution
- **Validation Tests**: System health checks with dependency verification

### Key Test Files
- `test_drift_detection.py`: ADWIN, MMD, Evidently detector validation
- `test_data.py`: Data loading and drift injection validation
- `test_server.py`: Server strategy and aggregation testing

### Test Execution
```bash
pytest tests/ -v           # Full test suite
python -m pytest tests/test_drift_detection.py -v  # Specific test file
```

## Hardware and Performance

### Supported Platforms
- **Primary**: macOS with Apple Silicon (MPS acceleration)
- **Secondary**: Intel macOS, Linux with CUDA/CPU

### Resource Requirements
- **Memory**: 8GB minimum, 16GB recommended for full simulation
- **Storage**: 10GB for datasets and model checkpoints
- **Compute**: Multi-core CPU, GPU optional but recommended

### Performance Optimization
- **Model**: BERT-tiny (4.4M parameters) for lightweight training
- **Batch Size**: 16 for client training, 8 for validation
- **Parallelization**: Ray-based Flower simulation with configurable resources

## Implementation Notes

### Flower Framework Integration
The system extends Flower's base classes:
- `DriftDetectionClient(NumPyClient)`: Adds drift monitoring to standard FL client
- `DriftAwareFedAvg(FedAvg)`: Adds mitigation logic to standard aggregation strategy

### Model Architecture
Uses `prajjwal1/bert-tiny` for computational efficiency:
- 4-class text classification (AG News dataset)
- 128 token max length
- Local SGD training with 3 epochs per round

### Drift Detection Pipeline
1. **Client Level**: ADWIN monitors performance metrics, Evidently analyzes data distributions
2. **Server Level**: MMD test analyzes embedding space drift across clients
3. **Aggregation**: Combined signals trigger mitigation strategy switch

### Error Handling and Logging
- Comprehensive logging with configurable levels (DEBUG, INFO, WARNING)
- Graceful fallbacks for missing dependencies or hardware
- Structured error reporting with actionable messages

## Experimental Workflow

### Standard Simulation Timeline
- **Rounds 1-24**: Stable FedAvg baseline training
- **Round 25**: Synthetic drift injection
- **Rounds 26-28**: Drift detection phase
- **Rounds 29-50**: Recovery with FedTrimmedAvg mitigation

### Key Metrics Tracked
- **Global Accuracy**: Weighted average across clients
- **Fairness Gap**: max(accuracy) - min(accuracy) across clients
- **Detection Delay**: Rounds until drift detection
- **Recovery Rate**: (Acc_recover - Acc_drift) / (Acc_base - Acc_drift)

### Expected Performance Targets
- Global Accuracy: >85% baseline, recovery to >80% of baseline
- Fairness Gap: <15% client disparity
- Detection Delay: ≤3 rounds after drift injection
- Recovery Rate: ≥80% within 10 rounds post-mitigation
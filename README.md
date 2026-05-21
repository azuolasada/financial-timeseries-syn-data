# Use of Synthetic Data in Financial Time Series Analysis

This repository contains the source code for a bachelor's thesis project investigating the impact of synthetic data on financial time series forecasting, specifically focusing on cryptocurrency markets.

## Project Overview

The project explores whether augmenting real financial datasets with synthetic data can improve the performance of deep learning models and mitigate overfitting. It compares three different synthetic data generation methods and evaluates their effectiveness across multiple prediction models and cryptocurrencies.

### Key Features

- **Data Extraction**: Automated fetching of historical cryptocurrency OHLCV data from Binance using `ccxt`.
- **Synthetic Data Generation**:
    - **Stationary Bootstrap (SB)**: A non-parametric resampling technique.
    - **QuantGAN**: A GAN-based approach designed for financial time series (Temporal Convolutional Networks).
    - **TC-VAE**: A Variational Autoencoder with Temporal Convolutions.
- **Prediction Models**:
    - **LSTM** (Long Short-Term Memory)
    - **BD-LSTM** (Bidirectional LSTM)
    - **TCN** (Temporal Convolutional Network)
- **Comprehensive Evaluation**: Tools to compare models trained on real data only versus models trained on mixed (real + synthetic) datasets with different weighting schemes.

## Repository Structure

- `src/`: Core source code.
    - `data/`: Data loading, extraction (from exchanges), and processing.
    - `generation/`: Implementations of synthetic data generators (QuantGAN, TC-VAE, SB).
    - `prediction/`: Prediction model architectures and training logic.
- `analysis/`: Jupyter notebooks for data analysis and result visualization.
- `data/`: Local storage for extracted and generated datasets (parquet format).
- `main.py`: Main entry point for running simulations.
- `config.yaml` / `config_syn.yaml`: Configuration files for experimental setups.

## Getting Started

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- `ccxt` for data extraction
- `pandas`, `numpy`, `pyyaml`, `scikit-learn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bsc-synthdata-crypto-overfit.git
   cd bsc-synthdata-crypto-overfit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Configure Experiments**: Modify `config.yaml` and `config_syn.yaml` to set your desired symbols, timeframes, date ranges, and model parameters.
2. **Run Simulations**: Execute the main script to start data extraction, generation, and model training:
   ```bash
   python main.py
   ```
3. **Analyze Results**: Explore the notebooks in the `analysis/` directory to visualize performance metrics and compare synthetic data methods.


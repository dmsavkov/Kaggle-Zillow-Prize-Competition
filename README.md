# Zillow 2016 Kaggle Competition

Source code for the Kaggle Zillow Prize competition (https://www.kaggle.com/competitions/zillow-prize-1).
Current status: IN ACTIVE DEVELOPMENT.

## Overview

This project implements a modular Python framework for analyzing real estate datasets, with focus on data quality assessment, feature engineering, and statistical testing. The codebase emphasizes reproducible analysis workflows and clean software architecture.

## Project Structure

```
zillow/                 # Core
├── analysis/
├── config/
├── features/
└── utils/

notebooks/             # Analysis workflows
scripts/               # Automation scripts
reports/               # Generated results and figures
```

## Key Components

### Statistical Analysis Suite

- **Drift Detection**: Implementation of KS tests, PSI, KL divergence, and Wasserstein distance
- **Hypothesis Testing**: Two-sample tests with proper multiple comparison corrections
- **Effect Size Calculations**: Cohen's d, Cliff's delta, and categorical association measures

### Data Quality Framework

- **Missing Data Analysis**: Correlation analysis and predictive modeling of missingness patterns
- **Adversarial Validation**: ML-based dataset similarity testing
- **Outlier Detection**: Systematic identification using statistical methods

### Feature Engineering Pipeline

- **Automated Processing**: Configurable data type conversion and transformation
- **Validation Workflows**: Cross-dataset consistency checks

## Analysis Results

The framework generates comprehensive reports including:

- Distribution comparisons between train/test datasets
- Feature importance rankings for missing data prediction
- Statistical drift detection across multiple metrics

## Technical Implementation

The project emphasizes modular and reproducable design, also following experiment tracking system (mlflow; not versioned) and CDD principles.

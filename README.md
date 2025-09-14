# Telematics Insurance POC

> **Usage-Based Insurance (UBI) system leveraging ML risk scoring and dynamic pricing for personalized auto insurance premiums**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org) [![Snowflake](https://img.shields.io/badge/Snowflake-Data%20Warehouse-blue.svg)](https://snowflake.com) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Status](https://img.shields.io/badge/Status-POC-orange.svg)]()

## Overview

This project implements a comprehensive telematics-based insurance system that uses real-time driving data, machine learning risk assessment, and dynamic pricing algorithms to create personalized insurance premiums. The system focuses on the Austin, Texas market and integrates real traffic incident data for enhanced accuracy.

### Key Features

- **Real-time Telematics Data Generation** - Synthetic driving data with Austin-specific geographic patterns
- **ML-Powered Risk Scoring** - Ensemble models (Random Forest, XGBoost, Neural Networks) for behavior analysis  
- **Dynamic Pricing Engine** - PAYD (Pay-As-You-Drive) and PHYD (Pay-How-You-Drive) models
- **Snowflake Integration** - Enterprise-scale cloud data warehouse storage
- **Austin Open Data API** - Real traffic incident integration for risk assessment
- **Comprehensive Analytics** - Driver profiling, risk categorization, and business metrics

### Business Impact

- **22% average savings** for safe drivers
- **2.8x premium differentiation** between risk levels  
- **Revenue-neutral design** with improved risk selection
- **68% R² model accuracy** for risk prediction

### Folder Structure

```austin-telematics-insurance/
├── bin/                                    # Executable scripts
│   └── create_script.sql                   # Database schema creation script
│
├── data/                                   # Generated data and sample outputs
│   ├── driver_risk_scores.csv              # ML-generated risk assessments per driver
│   ├── pricing_results.csv                 # Final premium calculations and comparisons
│   ├── sample_austin_incidents.csv         # Sample traffic incident data from Austin API
│   ├── sample_driver_profiles.csv          # Sample driver demographic and behavioral profiles
│   └── sample_telematics.csv               # Sample synthetic telematics data points
│
├── docs/                                   # Project documentation
│   └── Project Documentation.pdf           # Comprehensive technical and business documentation
│
├── models/                                 # Trained ML models
│   ├── risk_model_random_forest.pkl        # Random Forest risk scoring model
│   ├── risk_model_xgboost.pkl              # XGBoost risk scoring model  
│   ├── risk_model_neural_network.pkl       # Neural Network risk scoring model
│   ├── scaler_standard.pkl                 # StandardScaler for feature normalization
│   └── feature_names.pkl                   # Feature name mappings for model inference
│
├── src/                                    # Source code and main application files
│   ├── utils/                              # Utility functions and helper modules
|   |   ├── logging_config.py               # Configuration file for logging
│   ├── data_generator.py                   # Telematics data generation and Austin API integration
│   ├── pricing_engine.py                   # Dynamic pricing calculations (PAYD/PHYD models)
│   ├── risk_prediction.py                  # ML risk scoring pipeline and model training
│   └── app.py                              # Streamlit dashboard and web interface
│
├── .gitignore                              # Git ignore patterns for Python and data files
├── requirements.txt                        # Python package dependencies
├── .env.example                            # Environment variables template
└── README.md                               # Main project documentation and setup guide
```

## Quick Start

### Prerequisites

```bash
Python 3.8+
Snowflake account (optional - SQLite fallback available)
Austin Open Data API access
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/austin-telematics-insurance.git
cd austin-telematics-insurance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Snowflake credentials (optional)
```

### Basic Usage

1. **Generate Telematics Data**:
```bash
python src/data_generator.py
```

2. **Train Risk Scoring Models**:
```bash
python src/risk_prediction.py
```

3. **Calculate Dynamic Pricing**:
```bash
python src/pricing_engine.py
```

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Generator │────│   Snowflake     │────│  Risk Scoring   │
│  (Telematics)   │    │  Data Warehouse │    │  ML Pipeline    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│  Pricing Engine │──────────────┘
                        │   (PAYD/PHYD)   │
                        └─────────────────┘
```

## Component Details

### 1. Data Generation (`data_generator.py`)

Creates realistic telematics data with:
- **100 synthetic drivers** across 5 behavioral profiles
- **60 days** of driving history per driver  
- **Austin-specific geography** with real location mapping
- **Real traffic incidents** from Austin Open Data API
- **Quality validation** and anomaly detection

**Driver Profiles:**
- Safe (15% discount potential)
- Average (baseline)
- Aggressive (surcharge applicable)
- Young (higher risk patterns)
- Elderly (conservative patterns)

### 2. Risk Scoring (`risk_prediction.py`)

ML pipeline featuring:
- **Ensemble models** with overfitting prevention
- **7-component risk scoring**:
  - Speed violations (25% weight)
  - Hard braking events (20% weight)  
  - Phone usage (30% weight - critical factor)
  - Time-based risk (5% weight)
  - Acceleration patterns (15% weight)
  - Distance exposure (3% weight)
  - Incident proximity (2% weight)

**Model Performance:**
- Random Forest: R² = 0.65
- XGBoost: R² = 0.62
- Neural Network: R² = 0.45
- **Ensemble: R² = 0.68**

### 3. Dynamic Pricing (`pricing_engine.py`)

Implements sophisticated UBI models:

**PAYD (Pay-As-You-Drive) - 30% weight:**
- Low mileage (<8K annual): 15% discount
- High mileage (>20K annual): 20% surcharge

**PHYD (Pay-How-You-Drive) - 70% weight:**
- ML risk score integration
- Behavioral adjustments for specific violations
- Real-time incident risk factors

**Traditional Factors:**
- Age-based adjustments
- Vehicle year considerations
- Geographic risk factors

**Telematics Incentives:**
- 5% participation discount
- Up to 10% safe driver bonus
- 3% improvement rewards

## Data Pipeline

### Snowflake Tables

| Table | Purpose | Records |
|-------|---------|---------|
| `TELEMATICS_DATA` | Raw driving data points | ~700K |
| `DRIVER_PROFILES` | Driver characteristics | 100 |
| `AUSTIN_INCIDENTS` | Traffic incident data | ~400k |
| `RISK_SCORES` | ML-generated risk assessments | 100 |
| `PRICING_DATA` | Final premium calculations | 100 |

### Data Flow

```
Austin API → Data Generation → Feature Engineering → ML Training → Risk Scoring → Dynamic Pricing → Business Analytics
```

## Configuration

### Environment Variables

```env
# Snowflake Configuration (Optional)
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=AUSTIN_TELEMATICS
SNOWFLAKE_SCHEMA=RAW_DATA
SNOWFLAKE_ROLE=SYSADMIN
```

### Pricing Configuration

Key parameters in `PricingConfig`:
- `BASE_ANNUAL_PREMIUM = 1200` - Starting premium amount
- `MIN_RISK_MULTIPLIER = 0.65` - Maximum discount (35% off)
- `MAX_RISK_MULTIPLIER = 1.80` - Maximum surcharge (80% increase)

## Output Files

### Generated Data
- `data/austin_telematics.db` - SQLite database with all tables (not pushed because of file size limit)
- `data/sample_telematics.csv` - Sample telematics data
- `data/driver_risk_scores.csv` - ML-generated risk scores
- `data/pricing_results.csv` - Final premium calculations

### Model Artifacts
- `models/risk_model_*.pkl` - Trained ML models
- `models/scaler_*.pkl` - Feature scaling objects
- `models/feature_names.pkl` - Feature name mappings

### Logs
- `logs/telematics_generator.log` - Data generation logs
- Application logs for debugging and monitoring

## Sample Results

### Risk Score Distribution
- **Low Risk (35%)**: 0.0-0.37 score range
- **Medium Risk (45%)**: 0.37-0.53 score range  
- **High Risk (20%)**: 0.53-1.0 score range

### Premium Analysis
```
Traditional Premium Range: $950 - $1,450
Telematics Premium Range: $780 - $2,160
Average Savings: $220 (18% for safe drivers)
Revenue Impact: -1.8% (revenue neutral)
```

### Analytical Example
#### Top Performers
| Driver | Type | Risk Score | Savings | Premium |
|--------|------|------------|---------|---------|
| DRV_0042 | Safe | 0.23 | $495 (41%) | $705 |
| DRV_0018 | Safe | 0.28 | $420 (35%) | $780 |
| DRV_0091 | Elderly | 0.31 | $385 (32%) | $815 |

#### Highest Risk
| Driver | Type | Risk Score | Surcharge | Premium |
|--------|------|------------|-----------|---------|
| DRV_0067 | Aggressive | 0.89 | $540 (45%) | $1,740 |
| DRV_0023 | Young | 0.82 | $480 (40%) | $1,680 |
| DRV_0055 | Aggressive | 0.78 | $440 (37%) | $1,640 |

## Technology Stack

### Core Technologies
- **Python 3.8+** - Primary development language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **XGBoost** - Gradient boosting for advanced risk modeling
- **Snowflake** - Enterprise cloud data warehouse
- **Databricks** - Enterprise level data preprocessing
- **Apache Kafka** - Can be used for streaming telematic data
- **Apache Airflow** - For orchestration of data pipeline (ETL + Model training)
- **SQLite** - Local development and fallback storage

### ML & Analytics
- **Ensemble Methods** - Random Forest, XGBoost, Neural Networks
- **Feature Engineering** - Time-series, geospatial, behavioral patterns
- **Model Validation** - Cross-validation, holdout testing, business metrics
- **Risk Assessment** - Multi-component scoring with industry standards

### Data Integration  
- **Austin Open Data API** - Real-time traffic incident data
- **GeoPy** - Geographic calculations and distance computations
- **RESTful APIs** - External data source integration
- **Real-time Processing** - Streaming data pipeline capabilities

## API Integration

### Austin Open Data Integration

The system integrates live traffic incident data from Austin's open data platform:

```python
# Real-time incident data loading
url = "https://data.austintexas.gov/api/views/dx9v-zd7x/rows.json"
incidents = load_austin_incidents(url)

# Geographic risk factor calculation
for incident in incidents:
    risk_factor = calculate_proximity_risk(driver_location, incident_location)
    apply_risk_multiplier(risk_factor)
```

**Benefits:**
- Real-time risk adjustment based on current traffic conditions
- Geographic risk heat mapping across Austin
- Dynamic pricing updates for high-incident areas
- Historical pattern analysis for route risk assessment

## Model Validation

### Statistical Validation
- **Cross-Validation**: 5-fold CV with stratified sampling
- **Holdout Testing**: 35% test set for robust validation
- **Business Metrics**: Premium differentiation and category accuracy
- **Temporal Validation**: Time-series split for realistic performance

### Industry Standards Compliance
- **Actuarial Principles**: Risk factors must be statistically significant
- **Regulatory Requirements**: Model explainability and fairness testing
- **Anti-Discrimination**: Regular bias auditing across demographic groups
- **Consumer Protection**: Transparent risk factor communication

### Performance Monitoring
- **Model Drift Detection**: Automated monitoring of prediction quality
- **A/B Testing Framework**: Challenger model evaluation system
- **Business KPI Tracking**: Premium accuracy, loss ratios, customer satisfaction
- **Data Quality Monitoring**: Real-time data validation and anomaly detection

## Deployment Guide

### Production Checklist

**Infrastructure Requirements:**
- Snowflake data warehouse provisioned
- Python 3.8+ runtime environment
- SSL certificates for secure data transmission
- Backup and disaster recovery procedures

**Data Pipeline:**
- Real-time telematics data ingestion
- Automated data quality validation
- Model retraining scheduling (monthly)
- Performance monitoring dashboards

**Security & Compliance:**
- PII data encryption at rest and in transit
- Access control and audit logging
- Regulatory compliance validation
- Customer consent management system

**Business Integration:**
- Claims system data sharing
- Policy administration system integration
- Customer portal development  
- Agent training and support materials

### Scaling Considerations

**Current POC Capacity:**
- 100 drivers with 60 days of history
- ~700K telematics data points
- Real-time risk score updates
- Daily pricing recalculation

**Production Scale Targets:**
- 10,000+ active drivers
- 1M+ monthly telematics records
- Sub-second risk score computation
- Hourly pricing updates for high-risk changes


## License

This project is NOT licensed  - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Austin Open Data Initiative** for traffic incident data access
- **Snowflake** for cloud data warehouse platform
- **Scikit-learn Community** for machine learning framework
- **Insurance Industry Partners** for domain expertise and validation

## Contact

**Project Lead:** [Adarsh Kumar](mailto:adarsh0801@tamu.edu)  
**Organization:** Texas A&M University  
**LinkedIn:** [adarsh-k-tiwari](https://www.linkedin.com/in/adarsh-k-tiwari/)


---


*Permission is **NOT** granted to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of this software.*
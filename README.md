# Enterprise MLOps with Kubeflow + MLflow

<div align="center">
  <img src="https://img.shields.io/badge/Kubeflow-Orchestration-blue?style=for-the-badge" alt="Kubeflow" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-green?style=for-the-badge" alt="MLflow" />
  <img src="https://img.shields.io/badge/python-3.11.x-blue?style=for-the-badge" alt="Python Version" />
  <img src="https://img.shields.io/badge/status-Active-green?style=for-the-badge" alt="Status" />
</div>

<hr />


## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Kubeflow + MLflow Integration](#kubeflow--mlflow-integration)
- [Datasets](#datasets)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Setup & Installation](#setup--installation)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Kubeflow Configuration](#kubeflow-configuration)
  - [Executing Kubeflow Pipelines](#executing-kubeflow-pipelines)
- [ML Pipeline](#ml-pipeline)
- [MLflow Model Registry](#mlflow-model-registry)
- [Model Serving](#model-serving)
- [Frontend](#executing-frontend)
- [Troubleshooting](#troubleshooting)
- [Contribution](#contribution)


## Overview

This project is an **Enterprise-Grade Employee Attrition Prediction System** orchestrated with **Kubeflow** for workflow automation and **MLflow** for experiment tracking and model management. It predicts whether an employee will leave (attrition) or stay at a company based on various employee and workplace factors.

The system leverages Kubernetes-native orchestration (Kubeflow) to manage complex ML workflows with containerized components, while MLflow provides centralized experiment tracking, model versioning, and a model registry for production deployment.

**Business Value:**
- Identify employees likely to leave the organization using automated ML pipelines.
- Reduce turnover costs through intelligent workforce planning.
- Support HR decision-making with data-driven insights from tracked experiments.
- Ensure model governance with MLflow's centralized model registry.
- Scale ML workloads efficiently with Kubernetes orchestration.


## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubeflow Pipelines                      │
│  (Orchestration & Workflow Management on Kubernetes)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────────┐    ┌────────▼──────────────┐
│  Data Components     │    │  Model Components     │
│ - Ingestion          │    │ - Tuning              │
│ - Validation         │    │ - Training            │
│ - Cleaning           │    │ - Evaluation          │
│ - Feature Eng.       │    │ - Registration        │
│ - Preprocessing      │    │                       │
└───────────────────────┘    └──────────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    MLflow Integration                       │
│  (Experiment Tracking, Model Registry & Management)         │
├──────────────────────────────────────────────────────────────┤
│ • Track metrics, parameters & artifacts                      │
│ • Log model artifacts & metadata                             │
│ • Model versioning & staging                                 │
│ • Production model registry                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────────┐    ┌────────▼──────────────┐
│ KServe Inference     │    │ Flask Frontend        │
│ (Model Serving)      │    │ (Web UI)              │
└──────────────────────┘    └───────────────────────┘
```


## Kubeflow + MLflow Integration

### Why Kubeflow + MLflow?

**Kubeflow** provides:
- Kubernetes-native pipeline orchestration
- Containerized, reproducible component execution
- Automatic workflow scheduling and monitoring
- Scalable distributed training support

**MLflow** provides:
- Centralized experiment tracking and metrics visualization
- Model versioning and artifact management
- Production model registry for governance
- Model serving and API deployment

**Together they create:**
- A complete MLOps platform with separated concerns
- Reproducible pipelines with full observability
- Compliance-ready model governance
- Seamless transition from experimentation to production


## Datasets

The main dataset [employee_attrition.csv](./datasets/employee_attrition.csv) contains 74,500 employee records used across the Kubeflow pipeline. Data flows through each Kubeflow component, with processed artifacts stored at each stage in [datasets/data-pipeline/](./datasets).

**Dataset Features:**

| Feature | Description |
|---------|-------------|
| Employee ID | Unique identifier for each employee |
| Age | Age of the employee |
| Gender | Gender (Male/Female) |
| Years at Company | Tenure in years |
| Job Role | Position/Department |
| Monthly Income | Salary in monthly terms |
| Work-Life Balance | Rating (Poor/Good/Excellent) |
| Job Satisfaction | Level (Low/Medium/High) |
| Performance Rating | Employee performance (Low/Average/High) |
| Number of Promotions | Career progression count |
| Overtime | Whether employee works overtime (Yes/No) |
| Distance from Home | Commute distance in km |
| Education Level | Highest education attained |
| Marital Status | Single/Married/Divorced |
| Number of Dependents | Family dependents count |
| Job Level | Position hierarchy level |
| Company Size | Organization size (Small/Medium/Large) |
| Company Tenure | Time at current company |
| Remote Work | Remote work eligibility (Yes/No) |
| Leadership Opportunities | Career growth potential |
| Innovation Opportunities | Innovation involvement |
| Company Reputation | Market reputation |
| Employee Recognition | Recognition programs |
| **Attrition** | **Target: Stayed/Left** |
| dataset_type | Train/Test split indicator |

## Features

**Key Factors Influencing Employee Attrition:**
- **Compensation**: Monthly income, job level
- **Work Environment**: Work-life balance, overtime, remote work, distance from home
- **Career Development**: Promotions, leadership opportunities, innovation participation
- **Job Satisfaction**: Overall satisfaction, performance ratings, recognition
- **Demographics**: Age, tenure, marital status, education level
- **Company Factors**: Company size, reputation, tenure at organization


## Project Structure

```
employee-attrition-kubeflow/
|__ _feast/                             # feast setup
|  |__ feature_repo/
|    |__ feature_definitons.py
|     |__ feature_store.yaml
|
├── _kubeflow/                          # Kubeflow Pipeline & Components
│   ├── pipeline/
│   │   ├── full_pipeline.py            # Complete ML pipeline definition
│   │   └── submit_pipeline.py          # Submit pipeline to Kubeflow
│   └── components/
│       ├── data_components/            # Data processing components
│       │   ├── _01_ingestion.py        
│       │   ├── _02_validation.py       
│       │   ├── _03_cleaning.py         
│       │   ├── _04_feature_engg.py     
│       │   └── _05_preprocessing.py    
│       ├── model_components/           # Model training components
│       │   ├── _06_tuning.py           
│       │   ├── _07_training.py         
│       │   ├── _08_evaluation.py       
│       │   └── _09_register.py         
│       └── util/
│           └── wait_job.py             # Utility for pipeline job wait
│
├── _mlflow/                            # MLflow Integration
│   └── registry.py                     
│
├── src/                                # Standalone Python modules (local execution)
│   ├── data_preparation/
│   │   ├── _01_ingestion.py
│   │   ├── _02_validation.py
│   │   ├── _03_eda.py
│   │   ├── _04_cleaning.py
│   │   ├── _05_feature_engg.py
│   │   ├── _06_preprocessing.py
│   └── model_development/
│       ├── _07_tuning.py
│       ├── _08_training.py
│       ├── _09_evaluation.py
│       └── _10_registry.py
│       └── feast_sync.py
|
├── datasets/
│   ├── employee_attrition.csv          # Main dataset (74,500 records)
│   └── data-pipeline/                  # Intermediate processed datasets
├── frontend/                           # Flask Web UI
│   ├── app.py
│   ├── static/
│   └── templates/
│
├── notebook/
|__ inference.yaml                      # KServe setup
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── test.py                             # Test suite
├── Dockerfile                          # Container image for components
└── README.md
```

## Tech Stack

**Core MLOps:**
- **Kubeflow Pipelines**: Kubernetes-native ML workflow orchestration
- **Kubeflow Trainer Operator**: Kubernetes-native ML training system
- **Feast**: Feature Management, to ensure training and inference use same features
- **MLflow**: Experiment tracking, model versioning, and registry
- **KServe**: Model serving and inference (KServe deployment)

**ML & Data:**
- **Python Version**: 3.11+
- **scikit-learn**: Classification & hyperparameter tuning
- **pandas**: Data manipulation & processing
- **numpy**: Numerical computations
- **pandera**: Data schema validation
- **matplotlib, seaborn**: Data visualization

**Infrastructure:**
- **Kubernetes**: Container orchestration platform for Kubeflow
- **Docker**: Containerization for pipeline components
- **Flask, Flask-CORS**: Web framework for UI

**Data Serialization:**
- **Pickle (.pkl)**: Model artifact storage
- **JSON**: Configuration & metadata

**Development:**
- **Jupyter**: Notebook-based experimentation


## ML Pipeline

### Kubeflow Pipeline Components

The end-to-end ML workflow is orchestrated through Kubeflow as containerized, reusable components:

#### **Data Engineering Pipeline** (_kubeflow/components/data_components/)

1. **01_ingestion.py** - Load raw employee data from CSV/S3
   - Input: Raw dataset
   - Output: `minio://<ingestion.csv>`

2. **02_validation.py** - Validate data schema using Pandera
   - Input: `minio://<ingestion.csv>`
   - Output: `minio://<validation.csv>`
   - Validates: Data types, null values, ranges

3. **03_cleaning.py** - Handle missing values, outliers, inconsistencies
   - Input: `minio://<validation.csv>`
   - Output: `minio://<cleaned.csv>`
   - Techniques: Imputation, outlier detection, deduplication

4. **04_feature_engg.py** - Feature engineering & transformation
   - Input: `minio://<cleaned.csv>`
   - Output: `minio://<feature_engg.csv>`
   - Creates: New features, categorical encodings, interactions

5. **05_preprocessing.py** - Scaling, normalization, train-test split
   - Input: `minio://<feature_engg.csv>`
   - Output: `minio://<train.csv>`, `minio://<test.csv>`, `minio://<preprocessor.pkl>`
   - Applies: ColumnTransformer, test-train split (80-20)


#### **Feast Setup** (_kubeflow/components/model_components/)
   - Run feast to store datasets in both offline and online store


#### **Model Development Pipeline** (_kubeflow/components/model_components/)

6. **06_tuning.py** - Hyperparameter optimization
   - Input: Training data from preprocessing
   - Output: Best hyperparameters, tuning report
   - **Logged to MLflow**: Parameters, search space, CV scores
   - Methods: GridSearchCV, RandomSearchCV

7. **07_training.py** - Train classification models
   - Input: Preprocessed training data + tuned parameters
   - Output: Trained model artifacts
   - **Logged to MLflow**: 
     - Model artifacts (pickle)
     - Training metrics (accuracy, loss)
     - Model parameters
   - Models: Logistic Regression

8. **08_evaluation.py** - Evaluate on test set
   - Input: Trained model + test data
   - Output: Evaluation report
   - **Logged to MLflow**:
     - Evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
     - Confusion matrix
     - Feature importance
     - Cross-validation scores

9. **09_register.py** - Register best model in MLflow Registry
   - Input: Model artifacts + evaluation metrics
   - Output: Registered model with version & stage
   - **MLflow Operations**:
     - Register model in central registry
     - Set model stage (Staging/Production)

### MLflow Integration

**Experiment Tracking:**
- All training runs, metrics, and parameters are logged to MLflow
- Accessible via MLflow UI for comparison and analysis

**Model Registry:**
- Models automatically registered after evaluation
- Version control with metadata
- Stage management (Staging → Production)
- Artifact storage with full lineage

**Artifact Management:**
- Model pickles, preprocessors, and metadata stored
- Reproducible model loading for serving


### Accessing MLflow UI

For local setup:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Note**: Skip this step if MLflow is configured in your own cloud.

Navigate to `http://localhost:5000` to:
- View all experiment runs
- Compare model metrics and parameters
- Download model artifacts
- Track lineage and metadata


## Setup & Installation

### Prerequisites

- **Kubernetes Cluster**: Running 1.16+ (Minikube, EKS, GKE, or similar)
- **Kubeflow**: 1.6+ installed on cluster
- **Feast**: setup Postgres (for registry), redis (online store), S3/MinIO(offline store)
- **MLflow**: Accessible from cluster (local or remote server)
- **Docker**: For building component images
- **Python**: 3.11+
- **kubectl**: Configured to access your cluster

### Environment Setup

#### Step 1: Clone & Navigate Repository

```bash
git clone https://github.com/mlops-hub/kubeflow-training-pipeline.git
cd kubeflow-training-pipeline
```

#### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Kubeflow Configuration

#### Step 4: Configure Kubeflow Access

Set up kubeconfig to access your Kubeflow cluster:

```bash
# For local Minikube
kubectl config use-context minikube

# Or for cloud providers, update kubeconfig accordingly
```

#### Step 5: Configure MLflow

Set MLflow tracking server URI (environment variable or in code):

```bash
# Option 1: Environment variable
export MLFLOW_TRACKING_URI=http://localhost:5000
# or remote server:
# export MLFLOW_TRACKING_URI=https://mlflow.yourcompany.com

# Option 2: Update in pipeline configuration
# Edit _kubeflow/pipeline/full_pipeline.py and set MLflow URI
```

#### Step 6: Build Docker Images for Components

Components need to be containerized for Kubeflow execution:

```bash
# Build image with all dependencies
docker build -t <registry>/employee-attrition:latest .

# Push to your container registry
docker push <registry>/employee-attrition:latest
```

Update image references in `_kubeflow/pipeline/full_pipeline.py`:

```python
CONTAINER_IMAGE = "<registry>/employee-attrition:latest"
```

### Executing Kubeflow Pipelines

#### Option A: Run the Full ML Pipeline with Kubeflow

```bash
# Navigate to pipeline directory
cd _kubeflow/pipeline

# Submit pipeline to Kubeflow
python submit_pipeline.py
```

This will:
1. Load the pipeline definition from `full_pipeline.py`
2. Submit it to Kubeflow using the Kubeflow Pipelines SDK
3. Orchestrate all 9 components in sequence
4. Log all metrics/artifacts to MLflow
5. Register the best model in MLflow registry

**Monitor Pipeline Execution:**

```bash
# View pipeline status
kubectl get pods -n kubeflow-user-example-com

# Access Kubeflow Pipelines UI
# Visit: http://<kubeflow-url>:3000
```

#### Option B: Run Local Standalone Pipeline (No Kubernetes)

For local development/testing without Kubeflow:

```bash
cd src/data_preparation

# Sequential execution
python _01_ingestion.py
python _02_validation.py
python _03_eda.py
python _04_cleaning.py
python _05_feature_engg.py
python _06_preprocessing.py

cd ../../     # at root project
python -m src.model_development.feast_sync            # run feast
python -m src.model_development._07_tuning.py
python -m src.model_development._08_training.py          
python -m src.model_development._09_evaluation.py     
python -m src.model_development._10_registry.py       
```

#### Step 7: View MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access at: http://localhost:5000
```

In the MLflow UI, you can:
- View all experiment runs
- Compare model metrics and parameters
- Track model versions
- Access registered models in the registry

#### Step 8: Deploy Model with KServe (Optional)
Define Kserve using yaml script

```bash
inference.yaml
```

Run the yaml script in server
```bash
kubectl apply -f inference.yaml
```

To see service ready
```bash
kubectl get inferenceservices
```

To know about service in detail
```bash
kubectl describe <inference-pod-name>
```

## Executing Frontend

```bash
# Start Flask application
python -m frontend.app

# Access web UI at: http://localhost:5000
```

**Note:** Ensure model artifacts are available locally before starting the frontend, or configure it to fetch from MLflow registry.


## Troubleshooting

### Kubeflow Submission Issues

```bash
# Check pipeline submission logs
kubectl logs -f <pod-name> -n kubeflow-user-example-com

# Verify cluster access
kubectl cluster-info

# Check component images
docker images | grep employee-attrition
```

### MLflow Connection Issues

```bash
# Verify MLflow server is running
curl http://localhost:5000

# Check MLFLOW_TRACKING_URI environment variable
echo $MLFLOW_TRACKING_URI

```

## Contributing

Please read our [Contributing Guidelines](CONTRIBUTION.md) before submitting pull requests.

Contributions are welcome! Please follow standard Git workflow:
1. Create a feature branch (`git checkout -b feature/improvement`)
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

When contributing to the pipeline:
- Ensure Kubeflow components follow the component SDK standards
- Log metrics/artifacts to MLflow for reproducibility
- Update documentation for new pipeline components
- Test locally before submitting to Kubernetes

## License

This project is licensed under the [MIT License](LICENCE).

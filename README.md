# VKR Fraud System

### Hybrid Tabular–Graph System for Fraudulent Transaction Detection

**Bachelor Thesis (ВКР)**
*Bauman Moscow State Technical University — IU8 (Information Security)*

This repository contains a **reproducible machine learning system** for detecting fraudulent banking transactions using a **hybrid tabular–graph approach** combining gradient boosting and graph neural networks.

The project implements a **full ML pipeline** including:

* data preparation
* feature engineering
* tabular modeling
* graph construction
* graph neural networks
* model fusion
* decision policy
* evaluation
* automated report generation
* interactive graph visualization

The system is designed to resemble a **real anti-fraud scoring pipeline** used in financial institutions.

---

# Table of Contents

* Overview
* Research Contribution
* System Architecture
* Dataset
* Experimental Setup
* Installation
* Quick Start
* Pipeline Overview
* Project Structure
* Training Pipeline
* Evaluation & Decision Policy
* Additional Experiments
* Graph Visualization
* Outputs
* Reproducibility
* VKR Defense Demonstration

---

# Overview

Fraud detection in financial transactions is a challenging problem due to:

* **extreme class imbalance**
* **complex relational structures between entities**
* **evolving fraud patterns**

Traditional approaches rely only on **tabular features** extracted from transactions.
However, many fraud schemes involve **relational behavior** between accounts, devices, and identities.

This project implements a **hybrid system combining:**

| Component      | Model                                          |
| -------------- | ---------------------------------------------- |
| Tabular scorer | LightGBM                                       |
| Graph scorer   | Heterogeneous Graph Neural Network (GraphSAGE) |
| Fusion         | Logistic regression                            |

The system produces a **risk score for each transaction**, which is converted into decisions:

```
ALLOW / REVIEW / DENY
```

This mimics decision policies used in production anti-fraud systems.

---

# Research Contribution

The main contributions of this work are:

### Hybrid fraud detection architecture

A system combining:

* **tabular ML models**
* **graph neural networks**
* **meta-model fusion**

### Inductive graph inference

The graph model supports **inductive prediction** for new transactions not present in the training graph.

### Honest evaluation protocol

Evaluation uses **time-based splits** to avoid temporal leakage:

```
train → val → test
```

### Fusion strategy

A meta-model learns to combine tabular and graph predictions using **external validation data**.

### Additional experiments

The study includes:

* **GNN architecture ablation**
* **graph robustness experiments**
* **decision policy optimization**

### Practical anti-fraud system components

The project implements features typical for real fraud systems:

* decision thresholds
* investigation zones
* cost estimation
* case investigation tools

---

# System Architecture

```
Raw Transactions
       │
       ▼
Feature Extraction
       │
       ▼
Tabular Model (LightGBM)
       │
       ▼
Graph Construction
       │
       ▼
Graph Neural Network (GraphSAGE)
       │
       ▼
Fusion Model
       │
       ▼
Risk Score
       │
       ▼
Decision Policy
ALLOW / REVIEW / DENY
       │
       ▼
Investigation Tools
(Graph Visualization)
```

---

# Dataset

The system uses the public dataset:

**IEEE-CIS Fraud Detection (Kaggle)**

Dataset characteristics:

| Property     | Value             |
| ------------ | ----------------- |
| Transactions | ~590k             |
| Features     | ~430              |
| Fraud rate   | highly imbalanced |

Data sources include:

* transaction features
* card information
* device metadata
* email domains

Expected dataset location:

```
data/raw/ieee-cis/
  train_transaction.csv
  train_identity.csv
```

---

# Experimental Setup

### Time-based split

To simulate real deployment conditions:

| Split | Purpose               |
| ----- | --------------------- |
| train | model training        |
| val   | hyperparameter tuning |
| test  | final evaluation      |

### Metrics

Primary metric:

```
PR-AUC
```

Secondary metrics:

* ROC-AUC
* log-loss

PR-AUC is preferred due to **extreme class imbalance**.

---

# Installation

### Create environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements_torch_cpu.txt
pip install -r requirements_pyg_cpu.txt
pip install -r requirements.txt
```

---

# Quick Start

Run the entire pipeline with one command:

```bash
python -m scripts.run_all
```

Force recomputation:

```bash
python -m scripts.run_all --force
```

Run only evaluation and report:

```bash
python -m scripts.run_all --from-step A7_evaluate --to-step A11_auto_report
```

Final report:

```
reports/report.html
```

---

# Pipeline Overview

The training pipeline consists of the following stages.

```
Raw Data
   │
   ▼
Data Preparation
   │
   ▼
Tabular Model Training
   │
   ▼
Graph Construction
   │
   ▼
GNN Training
   │
   ▼
External Graph Prediction
   │
   ▼
Fusion Model
   │
   ▼
Evaluation
   │
   ▼
Report Generation
```

---

# Project Structure

```
vkr_fraud_system/

configs/
data/
artifacts/
reports/

src/fraud_system/

scripts/
```

Key components:

| Directory | Purpose                       |
| --------- | ----------------------------- |
| configs   | schema and configuration      |
| data      | raw and processed datasets    |
| artifacts | models and evaluation outputs |
| reports   | generated report              |
| scripts   | pipeline scripts              |

---

# Training Pipeline

The system implements the following training stages.

### A2 — Data preparation

```
scripts/00_prepare_data.py
```

* schema normalization
* merging identity features
* time split

Outputs:

```
data/processed/
data/splits/
```

---

### A3 — Tabular model

```
scripts/01_train_tabular.py
```

Model:

```
LightGBM
```

Outputs:

```
tabular model
PR curve
prediction files
```

---

### A4 — Graph construction

```
scripts/02_build_graph.py
scripts/03_make_graph_data.py
```

Graph contains nodes such as:

* transactions
* cards
* devices
* emails

---

### A5 — Graph neural network

```
scripts/04_train_gnn.py
```

Architecture:

```
Hetero GraphSAGE
(PyTorch Geometric)
```

---

### A10 — Fusion model

```
scripts/10_train_fusion_external.py
```

Meta-model:

```
Logistic Regression
```

Inputs:

```
tabular score
graph score
```

---

# Evaluation & Decision Policy

After scoring, transactions are assigned to zones:

| Score                     | Decision |
| ------------------------- | -------- |
| score < T_review          | ALLOW    |
| T_review ≤ score < T_deny | REVIEW   |
| score ≥ T_deny            | DENY     |

Thresholds are optimized using validation data.

Evaluation includes:

* PR curves
* zone distributions
* cost estimation

---

# Additional Experiments

### GNN Ablation

Architecture parameters tested:

* number of layers
* embedding size
* neighbors

Results stored in:

```
artifacts/evaluation/gnn_ablation.csv
```

---

### Graph Robustness

Experiment:

```
edge dropout
```

Purpose:

Evaluate model stability under graph perturbations.

Outputs:

```
graph_robustness.csv
graph_robustness_pr_auc_vs_drop.png
```

---

# Graph Visualization

The system includes an **interactive investigation tool**.

Generate ego-graph visualization:

```bash
python -m scripts.16_build_graph_viz
```

Output:

```
reports/assets/graph.html
```

This allows exploration of relationships between entities connected to suspicious transactions.

---

# Outputs

Main outputs:

```
artifacts/
reports/
```

Important files:

| File                            | Description             |
| ------------------------------- | ----------------------- |
| fusion_metrics_external.json    | final model metrics     |
| thresholds_fusion_external.json | decision thresholds     |
| report.html                     | final experiment report |

---

# Reproducibility

Key reproducibility features:

* deterministic pipeline
* one-click runner
* saved artifacts
* configuration files

Run the entire experiment:

```
python -m scripts.run_all
```

---

# VKR Defense Demonstration

For thesis defense it is recommended to show:

1️⃣ Final model results

```
fusion_external metrics
```

2️⃣ Decision zones

```
ALLOW / REVIEW / DENY
```

3️⃣ Automated report

```
reports/report.html
```

4️⃣ Graph investigation tool

```
reports/assets/graph.html
```

5️⃣ One-click pipeline

```
python -m scripts.run_all
```

---

# License

This repository was developed as part of a **Bachelor Thesis** at
**Bauman Moscow State Technical University**.

The code is provided for **research and educational purposes**.



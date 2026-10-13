# Diabetes Readmission Prediction Using Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/scikit--learn-MLPClassifier-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/Project-Educational%20Research-blue" alt="Project Status"/>
</p>

## Overview

This project implements an educational neural network workflow for multi-class hospital readmission prediction using the Diabetes 130-US Hospitals dataset.

The project bridges fundamental neural network concepts with a practical machine learning pipeline. It begins with manually implemented neurons and activation functions using NumPy, then progresses to data preprocessing, patient-level data splitting, neural network training, validation, and model artifact serialization.

The primary objective is to understand neural network mechanics and evaluate model behavior on an imbalanced healthcare classification problem.

**Current status:** Core neural network experiments, validation evaluation, class-balancing experiments, and model serialization have been implemented in Google Colab. Model optimization and final held-out test evaluation remain future work.

## Technical Objectives

- Implement a single neuron and a basic neural network layer using NumPy.
- Understand weighted sums, bias, forward propagation, ReLU, sigmoid, and squared error.
- Perform exploratory data analysis and assess missingness and target distribution.
- Build a preprocessing pipeline for numerical and categorical features.
- Prevent patient overlap between training, validation, and test partitions.
- Train a multi-class neural network using scikit-learn's `MLPClassifier`.
- Compare accuracy, macro F1-score, class-wise precision and recall, and confusion matrices.
- Investigate the impact of class imbalance and random oversampling.
- Serialize the model, preprocessing pipeline, and target encoder.

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

The dataset contains hospital encounter records for patients with diabetes.

| Attribute | Description |
|---|---|
| Original records | 101,766 encounters |
| Original columns | 50 |
| Prediction target | `readmitted` |
| Classification type | Multi-class classification |
| Target classes | `<30`, `>30`, `NO` |

The target categories represent readmission within 30 days, readmission after 30 days, and no recorded readmission, respectively, according to the dataset's labeling.

### Data Preparation

The implemented workflow includes:

- Replacing `?` markers with missing values.
- Analyzing missing-value counts and percentages.
- Examining class distribution and duplicate records.
- Excluding identifier columns from predictive features.
- Excluding `discharge_disposition_id` from the feature set in the current workflow.
- Splitting records by patient to avoid patient overlap across partitions.
- Imputing missing numerical values using the median.
- Imputing missing categorical values with a designated missing category.
- Standardizing numerical features.
- One-hot encoding categorical features.
- Encoding target labels for neural network training.

The fitted preprocessing pipeline is trained on the training partition and subsequently applied to validation and test data.

## Architecture and Methodology

### 1. Neural Network Fundamentals

The initial notebook sections demonstrate a neuron's weighted sum:

`z = w₁x₁ + w₂x₂ + b`

Activation functions introduce non-linearity into neural networks. The notebook explores ReLU and sigmoid before moving to a trainable classifier.

### 2. Classification Model

The main model uses scikit-learn's `MLPClassifier`.

Baseline configuration:

| Parameter | Value |
|---|---|
| Hidden layer sizes | `(64,)` |
| Activation | ReLU |
| Optimizer | Adam |
| Regularization (`alpha`) | `0.001` |
| Batch size | `256` |
| Initial learning rate | `0.001` |
| Maximum iterations | `30` |
| Random state | `42` |

Additional experiments investigate a deeper architecture, a lower learning rate, stronger regularization, early stopping, and oversampling.

### 3. Data Leakage Prevention

The notebook uses patient-level splitting rather than randomly splitting individual encounters. This reduces the risk of the same patient's records appearing in both training and evaluation partitions.

The dataset is partitioned into:

- Training subset for fitting the model.
- Validation subset for comparing experiments.
- Held-out test subset reserved for final evaluation.

The test partition should not be used for hyperparameter selection.

## Validation Results

The following results were observed during model experiments.

| Experiment | Validation Accuracy | Macro F1 |
|---|---:|---:|
| Baseline   64 neurons | 0.5274 | **0.4015** |
| Updated   128, 64 neurons | **0.5762** | 0.3916 |
| Fully balanced   128, 64 neurons | 0.4454 | 0.3818 |

*These are recorded intermediate validation results. They are not final test-set metrics.*

### Interpretation

- The updated model achieved the highest validation accuracy among these three experiments.
- The baseline achieved the highest macro F1-score.
- Full oversampling improved minority-class recall but reduced overall accuracy.
- The `<30` class remains difficult to identify reliably.
- The baseline reached its configured iteration limit without full convergence.

These results illustrate why model selection should not rely on accuracy alone. The appropriate model depends on the intended balance between overall performance and minority-class detection.

No clinical effectiveness or deployment readiness is claimed.

## Evaluation Metrics

The workflow uses the following evaluation measures:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the overall proportion of correct predictions. |
| Precision | Measures the correctness of class predictions. |
| Recall | Measures how many actual examples of a class are identified. |
| Macro F1-score | Gives each class equal weight in the overall F1 metric. |
| Confusion matrix | Shows correct and incorrect predictions for each class. |

For this imbalanced classification task, macro F1 and per-class recall are particularly important alongside accuracy.

## Technology Stack

- **Python**  implementation language
- **NumPy**   numerical computation and neuron fundamentals
- **Pandas**   tabular data preparation and analysis
- **Matplotlib**   activation-function plots and visualizations
- **SciPy**   sparse matrix operations
- **scikit-learn**   preprocessing, neural network training, and evaluation
- **Joblib**   serialization of trained artifacts
- **Google Colab / Jupyter Notebook**   interactive development environment

## Repository Structure

The intended repository structure is:

```text
diabetes-readmission-neural-network/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── introduction-to-neural-networks.ipynb
└── models/
    └── .gitkeep
```

The notebook filename and directory structure should be adjusted to match the actual repository files. Generated model artifacts may be stored separately rather than committed to Git.

## Setup and Execution

### Prerequisites

- Python 3.13 or a compatible Python environment.
- Access to the dataset from its official source.
- Jupyter Notebook or Google Colab.

### Install Dependencies

Create a `requirements.txt` file containing the packages used by the notebook, with versions aligned to the verified environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install the main dependencies directly:

```bash
pip install numpy pandas matplotlib scipy scikit-learn joblib jupyter
```

### Run the Notebook

1. Download the dataset from the UCI source.
2. Open the notebook in Google Colab or Jupyter.
3. Configure the dataset path according to the notebook's loading cell.
4. Execute the notebook cells in order.
5. Inspect training logs and validation metrics.
6. Verify saved model artifacts before using them elsewhere.

## Model Artifacts

The notebook currently generates these files:

| Artifact | Purpose |
|---|---|
| `best_neural_network.pkl` | Serialized baseline neural network |
| `preprocessor.pkl` | Fitted preprocessing pipeline |
| `label_encoder.pkl` | Target label encoder |

These artifacts must be loaded with compatible library versions and the expected input feature schema.

The filename `best_neural_network.pkl` identifies the selected baseline artifact in the current notebook; it does not imply that the model is fully optimized or clinically validated.

## Limitations and Risks

- The dataset contains historical records from 1999–2008.
- Class imbalance affects minority-class prediction quality.
- The baseline model did not converge within its configured iteration limit.
- Validation metrics may differ from final held-out test performance.
- Historical associations do not establish causation.
- Healthcare predictions can affect important decisions and require rigorous external validation.
- The current notebook is an educational prototype, not a production clinical decision-support system.

No patient-identifiable data should be committed to the repository. Dataset licensing and redistribution requirements should be checked before sharing data files.

## Roadmap

- [x] Implement basic neuron calculations with NumPy.
- [x] Explore ReLU and sigmoid activation functions.
- [x] Prepare the dataset and inspect class imbalance.
- [x] Implement patient-level training, validation, and test splitting.
- [x] Build the preprocessing pipeline.
- [x] Train and compare initial neural network configurations.
- [x] Experiment with random oversampling.
- [x] Save model and preprocessing artifacts.
- [ ] Complete systematic hyperparameter tuning.
- [ ] Select the final model using validation metrics.
- [ ] Evaluate the selected model once on the held-out test set.
- [ ] Add reproducible tests and environment documentation.
- [ ] Develop an optional Streamlit demonstration interface.

## License

Add an appropriate license file before distributing the project. An MIT license is one option if it matches the repository owner's intended terms and rights to the project code.

## Disclaimer

This project is for educational and research purposes only. It is not intended for diagnosis, treatment decisions, or clinical deployment. Any future real-world use would require independent validation, appropriate privacy safeguards, clinical oversight, and assessment of model fairness and reliability.

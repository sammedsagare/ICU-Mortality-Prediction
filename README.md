# ICU Mortality Prediction

A machine learning project for predicting ICU patient mortality risk using clinical and laboratory data. The model achieves an ROC-AUC of 0.80 on the test set using a calibrated Logistic Regression approach with SMOTE for handling class imbalance.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project develops a predictive model to assess mortality risk for ICU patients using 48 clinical features including demographics, vital signs, comorbidities, and laboratory values. The model helps healthcare providers identify high-risk patients for early intervention.

**Key Highlights:**
- **Model Type**: Calibrated Logistic Regression with SMOTE
- **Test ROC-AUC**: 0.800
- **Test PR-AUC**: 0.447
- **Optimal Threshold**: 0.46
- **Features Used**: 48 clinical parameters

## ✨ Features

- **Comprehensive Data Analysis**: Exploratory Data Analysis (EDA) with visualization
- **Advanced Preprocessing**: Handling missing values, outliers, and feature scaling
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Model Optimization**: Hyperparameter tuning with cross-validation
- **Probability Calibration**: Improved prediction reliability
- **Feature Importance Analysis**: Identification of key mortality predictors
- **Risk Stratification**: Patient categorization into risk groups

## 📊 Dataset

The dataset contains ICU patient records with:
- **Total Features**: 48
- **Feature Categories**:
  - Demographics: Age, Gender, BMI
  - Comorbidities: Hypertensive, Atrial Fibrillation, CHD, Diabetes, COPD, Renal Failure, etc.
  - Vital Signs: Heart Rate, Blood Pressure, Temperature, Respiratory Rate, SpO2
  - Laboratory Values: Blood counts, Electrolytes, Kidney function markers, Cardiac markers
  - Other Metrics: Urine output, Blood gases (pH, PCO2, Bicarbonate)

**Data Location**: `data/raw/dataset.csv`

## 📈 Model Performance

### Cross-Validation Metrics
- **ROC-AUC**: 0.811
- **PR-AUC**: 0.527
- **Recall**: 0.701

### Test Set Metrics
- **ROC-AUC**: 0.800
- **PR-AUC**: 0.447
- **Recall**: 0.344
- **Precision**: 0.688
- **F2-Score**: 0.382

### Optimal Hyperparameters
- **Regularization (C)**: 0.01
- **Penalty**: L2
- **SMOTE k_neighbors**: 5
- **SMOTE sampling_strategy**: 0.7

## 📁 Project Structure

```
ICU-Mortality-Prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Original dataset
│   └── processed/
│       ├── X_scaled.csv             # Preprocessed features
│       ├── y.csv                    # Target variable
│       └── top_features_for_ui.csv  # Top 12 features with coefficients
├── models/
│   ├── model_metadata.json          # Model performance & hyperparameters
│   └── feature_info.json            # Feature names and importance
└── notebooks/
    ├── 01_Data_Preprocessing_and_EDA.ipynb       # Data cleaning & exploration
    ├── 02_FE_and_Model_Training.ipynb            # Feature engineering & training
    ├── 03_SMOTE_RiskStrat_CV.ipynb               # SMOTE & risk stratification
    └── 04_Final_LR_Model.ipynb                   # Final model training
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Clone the Repository

```bash
git clone https://github.com/sammedsagare/ICU-Mortality-Prediction.git
cd ICU-Mortality-Prediction
```

### Create Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Data Preprocessing and EDA

Run the first notebook to explore and preprocess the data:

```bash
jupyter notebook notebooks/01_Data_Preprocessing_and_EDA.ipynb
```

This notebook covers:
- Data loading and inspection
- Missing value analysis
- Outlier detection and treatment
- Exploratory data analysis
- Statistical summaries and visualizations

### 2. Feature Engineering and Model Training

```bash
jupyter notebook notebooks/02_FE_and_Model_Training.ipynb
```

This notebook includes:
- Feature engineering and selection
- Data splitting (train/test)
- Feature scaling
- Initial model training
- Model evaluation

### 3. SMOTE and Risk Stratification

```bash
jupyter notebook notebooks/03_SMOTE_RiskStrat_CV.ipynb
```

This notebook performs:
- SMOTE implementation for class imbalance
- Risk stratification analysis
- Cross-validation
- Threshold optimization

### 4. Final Model Training

```bash
jupyter notebook notebooks/04_Final_LR_Model.ipynb
```

This notebook finalizes:
- Final Logistic Regression model
- Probability calibration
- Comprehensive evaluation
- Model persistence

### Running All Notebooks

To run all notebooks in sequence:

```bash
cd notebooks
jupyter notebook
```

Then execute notebooks in order (01 → 04).

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Imputation**: Mean/median imputation for numerical features
- **Outlier Treatment**: IQR-based outlier detection and capping
- **Feature Scaling**: StandardScaler normalization

### 2. Class Imbalance Handling
- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Sampling Strategy**: 0.7 (increasing minority class to 70% of majority)
- **k_neighbors**: 5

### 3. Model Selection
- **Algorithm**: Logistic Regression with L2 regularization
- **Calibration**: Probability calibration for reliable predictions
- **Regularization**: C=0.01 to prevent overfitting

### 4. Evaluation Strategy
- **Cross-Validation**: Stratified K-Fold
- **Metrics**: ROC-AUC, PR-AUC, Recall, Precision, F2-Score
- **Threshold Optimization**: Balanced precision-recall trade-off

## 🔑 Key Findings

### Top 12 Mortality Risk Predictors (by coefficient magnitude)

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | Renal failure | -0.325 | Protective |
| 2 | Deficiency anemias | -0.277 | Protective |
| 3 | Blood calcium | -0.238 | Protective |
| 4 | Urea nitrogen | +0.228 | Risk factor |
| 5 | Bicarbonate | -0.210 | Protective |
| 6 | Urine output | -0.199 | Protective |
| 7 | Leucocyte | +0.197 | Risk factor |
| 8 | Platelets | -0.196 | Protective |
| 9 | COPD | -0.194 | Protective |
| 10 | PCO2 | +0.194 | Risk factor |
| 11 | Creatinine | -0.169 | Protective |
| 12 | Heart rate | +0.159 | Risk factor |

**Note**: Negative coefficients indicate protective factors (lower mortality risk), while positive coefficients indicate risk factors (higher mortality risk).

### Clinical Insights
- **Kidney Function**: Multiple kidney-related markers (renal failure, urea nitrogen, creatinine) are strong predictors
- **Metabolic Factors**: Bicarbonate and blood calcium play significant roles
- **Inflammatory Response**: Leucocyte count is a key risk indicator
- **Cardiovascular**: Heart rate and PCO2 are important vital sign predictors
- **Hematological**: Platelets and anemia status affect outcomes

## 📦 Requirements

The project uses the following main libraries:

```
numpy                 # Numerical computing
pandas                # Data manipulation
matplotlib            # Visualization
seaborn               # Statistical visualization
plotly                # Interactive plots
scikit-learn          # Machine learning algorithms
imbalanced-learn      # SMOTE for class imbalance
xgboost               # Gradient boosting (exploration)
tensorflow            # Deep learning (exploration)
keras                 # Neural networks (exploration)
shap                  # Model interpretability
ipykernel             # Jupyter kernel
jupyter               # Notebook environment
```

Full dependencies listed in `requirements.txt`.

## 🔧 Model Files

The trained model metadata and feature information are stored in:
- `models/model_metadata.json`: Complete model performance metrics and hyperparameters
- `models/feature_info.json`: Feature names and importance rankings
- `data/processed/top_features_for_ui.csv`: Top features for UI/dashboard integration

<!-- ## 🎓 Future Improvements

- [ ] Implement ensemble methods (Random Forest, XGBoost)
- [ ] Deep learning approaches (Neural Networks)
- [ ] SHAP value analysis for model interpretability
- [ ] Web-based prediction interface
- [ ] Real-time prediction API
- [ ] Integration with electronic health records (EHR)
- [ ] Temporal analysis of patient trajectories
- [ ] External validation on different hospital datasets -->

## 📝 Notes

- The model was last trained on **October 26, 2025**
- **Random seed**: 42 (for reproducibility)
- **Optimal threshold**: 0.46 (can be adjusted based on clinical priorities)
- The model favors **recall** to minimize false negatives in high-risk scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch 
5. Open a Pull Request

## 📧 Contact

- Mail: [sammedsagare16@gmail.com](mailto:sammedsagare16@gmail.com)
- Repository: [ICU-Mortality-Prediction](https://github.com/sammedsagare/ICU-Mortality-Prediction)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This model is for research and educational purposes only. It should not be used as the sole basis for clinical decision-making without proper validation and oversight by qualified healthcare professionals.

---

**⭐ If you found this project helpful, please consider giving it a star!**

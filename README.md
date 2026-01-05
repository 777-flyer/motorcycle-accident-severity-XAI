# Motorcycle Accident Severity Prediction with Explainable AI

A comprehensive machine learning framework for predicting motorcycle accident severity with behavioral risk profiling and explainable AI techniques.

## Overview

This project implements seven machine learning models to predict motorcycle accident severity (No Accident, Moderate, Severe) using rider demographics, behavioral patterns, environmental conditions, and operational parameters. The framework achieves 95.3% accuracy with XGBoost and provides interpretable insights through SHAP and LIME explainability techniques.

## Key Features

- **High-Performance Prediction**: XGBoost model with 95.3% accuracy and 0.953 F1-score
- **Multi-Level Explainability**: SHAP for global feature importance and LIME for local instance explanations
- **Behavioral Risk Profiling**: K-Means clustering identifies four distinct rider profiles with severe accident rates ranging from 13.1% to 58.7%
- **Comprehensive Model Comparison**: Evaluation of 7 algorithms including XGBoost, Gradient Boosting, MLP, Random Forest, SVM, Decision Tree, and Logistic Regression

## Dataset

- **Source**: [Mendeley Data - Motorbike Accident Severity Analysis Dataset](https://data.mendeley.com/datasets/rm8t8kmhpj/3)
- **Size**: 15,100 motorcycle accident records from Bangladesh
- **Features**: 22 features spanning demographics, behavior, infrastructure, and environment
- **Classes**: Balanced distribution across No Accident (33.8%), Moderate Accident (35.4%), and Severe Accident (30.8%)

## Results

### Model Performance

| Model             | Accuracy  | Precision | Recall    | F1-Score  | ROC-AUC   |
| ----------------- | --------- | --------- | --------- | --------- | --------- |
| **XGBoost**       | **0.953** | **0.954** | **0.953** | **0.953** | **0.996** |
| Gradient Boosting | 0.950     | 0.950     | 0.950     | 0.950     | 0.996     |
| MLP               | 0.935     | 0.936     | 0.935     | 0.935     | 0.991     |
| Random Forest     | 0.929     | 0.931     | 0.929     | 0.929     | 0.992     |
| SVM               | 0.925     | 0.925     | 0.925     | 0.925     | 0.988     |

### Top Risk Factors (SHAP)

1. Number of Vehicles (0.842)
2. Talking While Riding (0.806)
3. Smoking While Riding (0.599)
4. Biker Occupation (0.304)
5. Daily Travel Distance (0.249)

### Rider Profiles

| Profile               | Size  | Avg Risk Score | Severe Accident Rate |
| --------------------- | ----- | -------------- | -------------------- |
| Safest Riders         | 23.1% | 0.69           | 13.1%                |
| Low-Risk Riders       | 20.4% | 0.92           | 19.8%                |
| High-Risk Riders      | 27.8% | 0.62           | 24.7%                |
| Severe Accident Prone | 28.7% | 2.61           | 58.7%                |

## Installation

```bash
# Clone repository
git clone https://github.com/777-flyer/motorcycle-accident-severity-XAI.git
cd motorcycle-accident-severity-XAI

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime
```

## Usage

```python
# Run the complete pipeline
python explainable-motorcycle-safety-ml.py

# Or open in Google Colab
# Upload the notebook and execute cells sequentially
```

## Project Structure

```
├── explainable-motorcycle-safety-ml.py    # python file
├── explainable-motorcycle-safety-ml.ipynb            # notebook
├── README.md
└── LICENSE
```

## Methodology

1. **Data Preprocessing**: Feature engineering, encoding, scaling with 80-20 stratified split
2. **Model Training**: 7 algorithms with hyperparameter tuning (GridSearchCV for XGBoost)
3. **Explainability**: SHAP for global importance, LIME for local explanations
4. **Behavioral Profiling**: K-Means clustering with PCA visualization

## Key Findings

- Distracted riding behaviors (talking, smoking) are the most critical modifiable risk factors
- Multi-vehicle scenarios dramatically increase accident severity
- 28.7% of riders fall into the "Severe Accident Prone" profile with 58.7% severe accident rate
- XGBoost achieves 23.7% improvement over logistic regression baseline

## Authors

- **Ahnaf Rahman Brinto**
- **Fayaz Bin Faruk**

Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Motorbike Accident Severity Analysis Dataset](https://data.mendeley.com/datasets/rm8t8kmhpj/3) by M. M. H. Matin
- Course: CSE427 Machine Learning, BRAC University
- Frameworks: scikit-learn, XGBoost, SHAP, LIME

---

**Note**: This project was developed as part of academic coursework.

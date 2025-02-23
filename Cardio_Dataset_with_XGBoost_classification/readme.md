# Cardiovascular Disease Prediction using XGBoost

This repository contains a machine learning project aimed at predicting cardiovascular disease using the XGBoost algorithm. The dataset used in this project is the `cardio_train.csv`(from Kaggle) dataset, which contains various health metrics such as age, weight, blood pressure, cholesterol levels, and more.

### Dataset Link : https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to predict the presence or absence of cardiovascular disease based on a set of health-related features. The project involves data preprocessing, feature selection, hyperparameter tuning, and model evaluation using the XGBoost classifier.

### Key Steps:
- **Data Loading and Preprocessing:** The dataset is loaded and preprocessed to handle missing values, remove outliers, and prepare it for model training.
- **Feature Selection:** Recursive Feature Elimination with Cross-Validation (RFECV) is used to select the most important features.
- **Model Training:** The XGBoost classifier is trained on the selected features.
- **Hyperparameter Tuning:** GridSearchCV is used to find the optimal hyperparameters for the XGBoost model.
- **Model Evaluation:** The model's performance is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## Dataset
The dataset used in this project is `cardio_train.csv`, which contains the following features:

- `id`: Unique identifier for each record.
- `age`: Age of the patient.
- `gender`: Gender of the patient.
- `height`: Height of the patient.
- `weight`: Weight of the patient.
- `ap_hi`: Systolic blood pressure.
- `ap_lo`: Diastolic blood pressure.
- `cholesterol`: Cholesterol level.
- `gluc`: Glucose level.
- `smoke`: Smoking status.
- `alco`: Alcohol consumption.
- `active`: Physical activity.
- `cardio`: Presence or absence of cardiovascular disease (target variable).

## Dependencies
To run this project, you need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Installation
Clone the repository:

```bash
git clone https://github.com/your-username/cardiovascular-disease-prediction.git
```

Navigate to the project directory:

```bash
cd cardiovascular-disease-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Ensure that the `cardio_train.csv` dataset is in the same directory as the Jupyter notebook.

Open the Jupyter notebook:

```bash
jupyter notebook cardiovascular_disease_prediction.ipynb
```

Run the notebook cells sequentially to load the data, preprocess it, train the model, and evaluate its performance.

## Results
The XGBoost model achieved an accuracy of **73%** on the test set. The classification report and confusion matrix are as follows:

### Classification Report
```
              precision    recall  f1-score   support

           0       0.71      0.79      0.74      5675
           1       0.75      0.66      0.70      5465

    accuracy                           0.73     11140
   macro avg       0.73      0.72      0.72     11140
weighted avg       0.73      0.73      0.72     11140
```

### Confusion Matrix
```
[[4462 1213]
 [1849 3616]]
```

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
code explanation

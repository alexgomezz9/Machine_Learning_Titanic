# # Kaggle Titanic - Machine Learning from Disaster

This repository contains the work done for the **Kaggle Titanic: Machine Learning from Disaster** competition. The goal is to predict which passengers survived the Titanic shipwreck based on various features such as age, sex, ticket class, and others.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Models Used](#models-used)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is focused on predicting the survival of passengers aboard the Titanic. The dataset includes information such as passenger age, gender, ticket class, and whether they survived or not. The main goal is to build a machine learning model that can predict whether a passenger survived based on these features.

### Key steps:
- Data cleaning and handling missing values.
- Feature engineering: creating new features from the available data.
- Exploratory data analysis to identify important patterns.
- Model selection: testing various classification models.
- Hyperparameter tuning using **RandomizedSearchCV** and **GridSearchCV**.
- Final model trained to predict survival outcomes.

## Data

The dataset used in this project is provided by Kaggle for the Titanic competition and can be found here: https://www.kaggle.com/c/titanic

The dataset includes:
- **train.csv**: Training data with features and the target (`Survived`).
- **test.csv**: Test data without labels (you need to predict `Survived`).

### Key features used for prediction:
- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Fare**: Passenger fare.
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

## Models Used

We experimented with various classification models to predict survival, including:

1. **Logistic Regression**: A linear model for binary classification.
2. **Random Forest Classifier**: An ensemble model that creates multiple decision trees and averages their predictions.
3. **Gradient Boosting Classifier**: A boosting technique that sequentially improves the model by minimizing errors from previous iterations.

### Hyperparameter Tuning

We used **RandomizedSearchCV** and **GridSearchCV** to fine-tune hyperparameters for the best performance on the validation set. Some of the tuned parameters include:
- `n_estimators` for ensemble models.
- `max_depth` of the trees.
- `learning_rate` for gradient boosting models.

## Results

The final model used is **GradientBoostingClassifier**, which provided the best accuracy in predicting passenger survival.

Final **Accuracy** on validation set: **0.82**

## Usage

To train the model and generate predictions for the test set, follow these steps:

1. Clone this repository by copying the URL of your repository and using the git clone command.

2. Run the data preprocessing and model training script by executing the appropriate Python files for preprocessing and training.

The predictions will be saved in a file named `titanic_predictions_submission.csv`.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. 


# Credit Card Fraud Detection Using Machine Learning and Transactional Data
### Overview
Credit card fraud is a growing concern, and detecting fraudulent transactions efficiently is crucial for maintaining the security and trust of financial institutions and customers. In this project, we use machine learning algorithms to detect patterns in real time using transaction data that indicate potential fraud.

This project builds and evaluates an end-to-end fraud detection pipeline using transactional data, with emphasis on feature engineering, class imbalance, and model evaluation beyond accuracy. 
Rather than treating fraud detection as a standalone model, the system is designed as a scalable decision-making process, highlighting trade-offs between precision, recall, and operational risk. The goal is to demonstrate how machine learning supports reliable, high-volume financial systems.

### Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. This dataset consists of credit card transactions made by cardholders, and the features include:

- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1, V2, ..., V28`: 28 anonymized features created through PCA (Principal Component Analysis) transformation.
- `Amount`: The amount of the transaction.
- `Class`: The target variable (0 for legitimate, 1 for fraudulent).

### Software
Python
Interface: Jupyter Notebooks.

### Data ceaning and validation
- Data cleaning was conductecd by handling mising values.
- Data validation was conducted by checking for accuracy of the dataset.

### Data modelling and processing
 Data Preprocessing was done by handling missing values, normalization, and encoding categorical features.
The data was then split into training and test sets.
Model Selection:
  -  Several machine learning models were tested, including Logistic Regression and Random Forest
    
  - The models were trained and evaluated based on accuracy, precision, recall, and F1-score.
### Model Evaluation & refinement:
   - Due to class imbalance the focus was on precsion & recall rather than acccuracy.
   - The model was evaluated using a confusion matrix to calculate the precision, recall, and F1-score, as fraudulent transactions  require special attention  due to cases of false positives and false negatives.

## Results

The model's performance is evaluated using the following metrics:

- **Accuracy**: The percentage of correct predictions
- **Precision**: The percentage of predicted fraudulent transactions that are actually fraudulent.
- **Recall**: The percentage of actual fraudulent transactions that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.








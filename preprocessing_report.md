# Data Preprocessing Report

**Project:** Customer Churn Prediction

**Dataset:** customer_churn.csv

**Rows:** 500

**Columns:** 9 (original dataset)

---

# 1. Project Objective

The objective of this project is to build a machine learning model capable of predicting **customer churn**, which refers to customers discontinuing their service. Accurate churn prediction allows businesses to proactively retain customers through targeted strategies.

To achieve reliable predictions, the dataset must undergo several preprocessing steps including:

* Data cleaning
* Handling categorical variables
* Feature scaling
* Outlier detection
* Feature engineering
* Feature selection
* Pipeline creation

These steps ensure the dataset is suitable for machine learning algorithms.

---

# 2. Dataset Overview

The dataset contains customer information related to subscription behavior.

| Column           | Description                                 |
| ---------------- | ------------------------------------------- |
| CustomerID       | Unique identifier for each customer         |
| Tenure           | Number of months the customer has stayed    |
| MonthlyCharges   | Monthly subscription charges                |
| TotalCharges     | Total amount charged to the customer        |
| Contract         | Contract type                               |
| PaymentMethod    | Payment method used                         |
| PaperlessBilling | Whether the customer uses paperless billing |
| SeniorCitizen    | Indicates if customer is a senior citizen   |
| Churn            | Target variable (1 = churn, 0 = retained)   |

The **Churn column** is the target variable used for prediction.

---

# 3. Data Cleaning

Initial exploration was performed to understand the structure of the dataset.

### Steps Performed

* Checked dataset shape and structure
* Inspected column data types
* Verified missing values
* Examined distribution of the churn variable

Example code:

```python
df.info()
df.describe()
df.isnull().sum()
```

No significant missing values were found in the dataset.

---

# 4. Handling Categorical Data

Machine learning algorithms require numerical input. Therefore categorical variables were converted into numerical format.

Three encoding methods were implemented.

### 4.1 Label Encoding

Used for ordinal or limited categorical features.

Example:

```python
LabelEncoder()
```

Applied to:

* Contract
* PaymentMethod

---

### 4.2 One-Hot Encoding

One-Hot Encoding converts categorical values into binary columns.

Example:

```python
pd.get_dummies()
```

Generated features such as:

* Contract_One year
* Contract_Two year
* PaymentMethod_Credit Card
* PaymentMethod_Electronic Check

---

### 4.3 Binary Encoding

Binary mapping was applied to boolean variables.

Example:

```python
df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes":1,"No":0})
```

---

# 5. Feature Scaling

Feature scaling was applied to normalize numerical features and improve model performance.

Two scaling methods were implemented.

### 5.1 Standard Scaling

Standardization transforms features so they have:

* Mean = 0
* Standard deviation = 1

Example:

```python
StandardScaler()
```

---

### 5.2 Min-Max Scaling

Min-Max scaling normalizes features to a fixed range.

Range used:

```
0 to 1
```

Example:

```python
MinMaxScaler()
```

---

# 6. Outlier Detection and Handling

Outliers can negatively affect machine learning models.

Two statistical methods were used to detect outliers.

### 6.1 Interquartile Range (IQR)

The IQR method identifies extreme values using quartiles.

Formula:

```
IQR = Q3 − Q1
```

Outlier range:

```
Lower bound = Q1 − 1.5 * IQR  
Upper bound = Q3 + 1.5 * IQR
```

---

### 6.2 Z-Score Method

Z-Score measures how far a data point is from the mean.

```
Z > 3 → potential outlier
```

Detected outliers were handled using **capping techniques**.

---

# 7. Feature Selection

Feature selection was performed to identify the most relevant predictors.

Two approaches were used.

### 7.1 Correlation Analysis

Correlation heatmaps were used to analyze relationships between variables.

Example:

```python
sns.heatmap(df.corr())
```

---

### 7.2 Feature Importance

A Random Forest model was used to determine feature importance.

This helped identify variables with the strongest influence on churn prediction.

---

# 8. Data Preprocessing Pipeline

A complete preprocessing pipeline was implemented using **Scikit-Learn Pipeline** and **ColumnTransformer**.

Pipeline stages include:

1. Feature scaling
2. Feature transformation
3. Model training

Example structure:

```python
Pipeline([
 ("preprocessor", preprocessor),
 ("model", LogisticRegression())
])
```

The pipeline ensures reproducibility and allows preprocessing and modeling to be executed in a single workflow.

---

# 9. Final Dataset

After preprocessing and feature engineering, the dataset contained additional derived features used for model training.

The processed dataset was used to train a **Logistic Regression model** for churn prediction.

---

# 10. Conclusion

Data preprocessing significantly improves machine learning model performance by:

* Cleaning inconsistent data
* Converting categorical variables
* Scaling numerical features
* Detecting and handling outliers
* Preparing features for model training

A complete preprocessing pipeline was successfully implemented to prepare the dataset for churn prediction.

---

# Tools and Libraries Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

![Progress](https://img.shields.io/badge/Progression-100%25-green)

# TITANIC-SURVIVE-PREDICTION-USING-EDA-AND-NAIVE-BAYES-CLASSIFIERS

📘 Overview
This project analyzes the Titanic dataset to predict the survival of passengers based on different features such as age, gender, class, and embarkation point.  
It combines *Exploratory Data Analysis (EDA)* and *Naive Bayes Classifier* to understand data patterns and build a predictive machine learning model.

---

## 🎯 Objective
The main goal of this project is to:
•⁠  ⁠Perform *Exploratory Data Analysis* to identify trends and correlations.
•⁠  ⁠Clean and preprocess the dataset for modeling.
•⁠  ⁠Apply *Naive Bayes Classification* to predict survival chances.
•⁠  ⁠Evaluate the model performance using accuracy and other metrics.

---

## 🧠 Machine Learning Algorithm Used
### 🔹 Naive Bayes Classifier
Naive Bayes is a simple yet powerful probabilistic classifier based on Bayes' Theorem.  
It assumes independence between features and is widely used for classification tasks such as spam detection, sentiment analysis, and survival prediction.

Mathematical formula:
\[
P(Y|X) = \frac{P(X|Y) * P(Y)}{P(X)}
\]

---

## 🧩 Steps Followed in the Project

### 1️⃣ Importing Libraries
Imported essential libraries such as:
•⁠  ⁠⁠ pandas ⁠ for data handling  
•⁠  ⁠⁠ numpy ⁠ for numerical operations  
•⁠  ⁠⁠ matplotlib ⁠ and ⁠ seaborn ⁠ for visualization  
•⁠  ⁠⁠ sklearn ⁠ for model training and evaluation  

### 2️⃣ Loading the Dataset
Loaded the Titanic dataset (usually from Kaggle) using Pandas and displayed the first few rows to understand its structure.

### 3️⃣ Exploratory Data Analysis (EDA)
Performed data visualization and statistical exploration to understand:
•⁠  ⁠Age and gender distribution  
•⁠  ⁠Survival rates by gender and passenger class  
•⁠  ⁠Correlation between features  

### 4️⃣ Data Cleaning & Preprocessing
•⁠  ⁠Handled missing values in *Age, **Cabin, and **Embarked* columns.  
•⁠  ⁠Converted categorical variables (like ⁠ Sex ⁠ and ⁠ Embarked ⁠) into numeric form using *Label Encoding* or *One-Hot Encoding*.  
•⁠  ⁠Removed irrelevant columns like ⁠ Name ⁠, ⁠ Ticket ⁠, and ⁠ Cabin ⁠.

### 5️⃣ Feature Selection
Selected key predictive features such as:
⁠ Pclass ⁠, ⁠ Sex ⁠, ⁠ Age ⁠, ⁠ SibSp ⁠, ⁠ Parch ⁠, and ⁠ Embarked ⁠.

### 6️⃣ Splitting the Dataset
Divided the data into:
•⁠  ⁠*Training set* – 80%
•⁠  ⁠*Testing set* – 20%

Using:
⁠ python
from sklearn.model_selection import train_test_split

7️⃣ Model Training
Trained the Naive Bayes classifier (GaussianNB or MultinomialNB) using:
from sklearn.naive_bayes import GaussianNB

8️⃣ Model Evaluation
Evaluated the model using:
Accuracy Score
Confusion Matrix
Classification Report
Example:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

9️⃣ Results
Achieved accuracy around 75–82% depending on preprocessing.
The model showed that gender and passenger class were the strongest survival predictors.


🔟 Visualization
Used seaborn and matplotlib to plot:
Count plots for survivors vs non-survivors
Heatmap of feature correlation
Age distribution by survival status


📊 Technologies Used
Category
Tools / Libraries
Language
Python
Data Analysis
Pandas, NumPy
Visualization
Matplotlib, Seaborn
Machine Learning
Scikit-learn
Notebook
Jupyter Notebook (.ipynb)


🗂️ Project Structure


├── TITANIC SURVIVE PREDICTION USING EDA AND NAIVE BAYES CLASSIFIERS.ipynb
├── README.md
├── .gitignore
└── LICENSE


🚀 How to Run
Clone the repository:
Bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
Open the Jupyter Notebook:
jupyter notebook "TITANIC SURVIVE PREDICTION USING EDA AND NAIVE BAYES CLASSIFIERS.ipynb"
Run all cells in order.
View predictions and visualizations.


🧩 Future Work
Experiment with other algorithms (Logistic Regression, Decision Tree, Random Forest)
Tune hyperparameters for better accuracy
Deploy model using Flask or Streamlit


👨‍💻 Author


Brijesh Rath


📧 Email: rathbrijesh2006@gmail.com


💼 GitHub: (https://github.com/Brijeshrath67)


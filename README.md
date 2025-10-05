# Coding-Samurai-Internship
# Project 1
# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

##  Project Overview
This project performs an **Exploratory Data Analysis (EDA)** on the classic **Titanic dataset** — one of the most well-known datasets in data science.  
The goal is to analyze the characteristics of passengers aboard the Titanic and uncover key factors that influenced their survival chances.  

By performing detailed data exploration, visualization, and interpretation, this project provides a structured understanding of how data-driven insights can be extracted from raw datasets.

---

## 🎯 Objective
The main objective is to:
- Explore and understand the Titanic dataset structure.
- Identify patterns and correlations affecting passenger survival.
- Handle missing data and visualize meaningful trends.
- Build a foundation for future predictive modeling tasks.

---

##  Skills and Tools Used
| Category | Tools & Libraries |
|-----------|-------------------|
| **Programming Language** | Python |
| **Libraries** | Pandas, NumPy, Matplotlib, Seaborn |
| **Environment** | Google Colab |
| **Concepts Applied** | Data Cleaning, Missing Value Handling, Visualization, Statistical Analysis |

---

##  Project Workflow and Step-by-Step Explanation

### **Step 1: Importing Essential Libraries**
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
## Explanation:
The project begins by importing essential Python libraries:

- **Pandas** for data manipulation and exploration.

- **NumPy** for numerical operations.

- **Matplotlib** and **Seaborn** for visualization.
These tools collectively provide a strong foundation for analyzing structured datasets.

### **Step 2: Loading the Dataset**
```python
import seaborn as sns
titanic = sns.load_dataset('titanic')
```
## Explanation:
We use Seaborn’s built-in Titanic dataset, which contains passenger details such as age, class, gender, and survival status.
This dataset is widely used to demonstrate data analysis techniques.
Loading the dataset successfully is the first step before understanding its structure.

### **Step 3: Understanding the Dataset**
```python
titanic.shape
titanic.info()
titanic.head()
titanic.describe()
```
### Explanation:

- .shape shows the number of rows and columns — representing total passengers and features.

- .info() provides data types and null value information.

- .head() previews the first few rows to get a sense of the data.

- .describe() gives summary statistics for numeric columns such as mean, min, max, and standard deviation.

This step establishes a baseline understanding of what data we’re dealing with.

### **Step 4: Checking for Missing Values**
```python
titanic.isnull().sum()
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
```
### Explanation:
Missing data is common in real-world datasets.

- .isnull().sum() counts how many values are missing per column.

- The **heatmap** visually highlights where missing values exist.
This helps determine whether to fill or drop missing data based on its impact and volume.

### **Step 5: Data Cleaning**
```python
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic.drop(['deck'], axis=1, inplace=True)
```
### Explanation:

- Missing ages are replaced with the median to retain valuable records without distorting the data.

- The deck column, which contains too many missing values, is dropped entirely.
This step ensures a cleaner, more reliable dataset for analysis.

## **Step 6: Univariate Analysis**
### a. Age Distribution
```python
sns.histplot(titanic['age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution of Passengers")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%203.png)
### Interpretation:
The histogram shows that most passengers were between **20–40 years old**, suggesting that the majority on board were young adults.

### b. Passenger Class Distribution
```python
sns.countplot(x='class', data=titanic)
plt.title("Passenger Count by Class")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%204.png)
### Interpretation:
Most passengers were in **3rd class**, indicating that the ship had a higher number of lower-income travelers.

### c. Gender Distribution
```python
sns.countplot(x='sex', data=titanic)
plt.title("Gender Distribution of Passengers")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%205.png)
### Interpretation:
There were **more males** than females on board, which may influence overall survival statistics.

### **Step 7. Bivariate Analysis** (Relationships between two variables)
### a. Survival by Gender
```python
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title("Survival Count by Gender")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%206.png)
### Interpretation:
A much higher percentage of **females survived** compared to males — consistent with the “women and children first” policy during evacuation.

### b. Survival Rate by Class
```python
sns.barplot(x='class', y='survived', data=titanic)
plt.title("Survival Rate by Passenger Class")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%207.png)
### Interpretation:
Passengers in **1st class** had a significantly higher survival rate than those in 2nd or 3rd class — showing that **social and economic status** played a role in survival chances.

### c. Age vs Fare Colored by Survival
```python
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic)
plt.title("Age vs Fare (Colored by Survival)")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%208.png)
### Interpretation:
Passengers who paid higher fares (mostly older and from 1st class) had **better survival rates**. Younger passengers in lower fare categories were less likely to survive.

### **Step 9: Correlation Analysis**
```python
corr = titanic.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%209.png)
### Interpretation:
The heatmap shows numeric correlations:

- **Fare has a positive correlation with Survival, meaning higher fares increased survival likelihood**.

- **Pclass is negatively correlated with Survival, confirming that lower classes faced higher death rates**.

# Key Insights & Findings
##🔍 Insights Summary

- **1. Gender: Females had a much higher survival rate than males.**

- **2. Class: Passengers in 1st class had a significantly better chance of survival.**

- **3. Fare: Higher fares correlated positively with survival (wealthier passengers survived more).**

- **4. Age: Younger passengers showed slightly higher survival chances, though the relationship wasn’t strong.**

- **5. Missing Data: Some columns like deck were mostly empty, so they were dropped to maintain accuracy.**
  # 🧾 Conclusion

The EDA reveals clear **social and economic disparities** in survival rates aboard the Titanic. Factors such as **gender, class, and fare** were strong determinants of survival, reflecting the societal norms of the early 1900s.

By performing this analysis, we not only understand the dataset’s structure but also see how real-world inequalities manifested in survival outcomes.

This EDA serves as a foundational step before building predictive models, as it highlights which features most strongly influence survival probability.

## 💡 Future Work

- Build a predictive model to estimate survival chances.

- Apply logistic regression or decision trees using the cleaned dataset.

- Extend analysis with additional visualizations for deeper insights.

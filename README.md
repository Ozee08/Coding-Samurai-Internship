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

## **Step 1: Importing Essential Libraries**
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

## **Step 2: Loading the Dataset**
```python
import seaborn as sns
titanic = sns.load_dataset('titanic')
```
## Explanation:
We use Seaborn’s built-in Titanic dataset, which contains passenger details such as age, class, gender, and survival status.
This dataset is widely used to demonstrate data analysis techniques.
Loading the dataset successfully is the first step before understanding its structure.

## **Step 3: Understanding the Dataset**
```python
titanic.shape
titanic.info()
titanic.head()
titanic.describe()
```
## Explanation:

- .shape shows the number of rows and columns — representing total passengers and features.

- .info() provides data types and null value information.

- .head() previews the first few rows to get a sense of the data.

- .describe() gives summary statistics for numeric columns such as mean, min, max, and standard deviation.

This step establishes a baseline understanding of what data we’re dealing with.

## **Step 4: Checking for Missing Values**
```python
titanic.isnull().sum()
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
```
## Explanation:
Missing data is common in real-world datasets.

- .isnull().sum() counts how many values are missing per column.

- The **heatmap** visually highlights where missing values exist.
This helps determine whether to fill or drop missing data based on its impact and volume.

## **Step 5: Data Cleaning**
```python
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic.drop(['deck'], axis=1, inplace=True)
```
## Explanation:

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
## Interpretation:
The histogram shows that most passengers were between **20–40 years old**, suggesting that the majority on board were young adults.

### b. Passenger Class Distribution
```python
sns.countplot(x='class', data=titanic)
plt.title("Passenger Count by Class")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%204.png)
## Interpretation:
Most passengers were in **3rd class**, indicating that the ship had a higher number of lower-income travelers.

### c. Gender Distribution
```python
sns.countplot(x='sex', data=titanic)
plt.title("Gender Distribution of Passengers")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%205.png)
## Interpretation:
There were **more males** than females on board, which may influence overall survival statistics.

## **Step 7. Bivariate Analysis** (Relationships between two variables)
### a. Survival by Gender
```python
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title("Survival Count by Gender")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%206.png)
## Interpretation:
A much higher percentage of **females survived** compared to males — consistent with the “women and children first” policy during evacuation.

### b. Survival Rate by Class
```python
sns.barplot(x='class', y='survived', data=titanic)
plt.title("Survival Rate by Passenger Class")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%207.png)
## Interpretation:
Passengers in **1st class** had a significantly higher survival rate than those in 2nd or 3rd class — showing that **social and economic status** played a role in survival chances.

### c. Age vs Fare Colored by Survival
```python
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic)
plt.title("Age vs Fare (Colored by Survival)")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%208.png)
## Interpretation:
Passengers who paid higher fares (mostly older and from 1st class) had **better survival rates**. Younger passengers in lower fare categories were less likely to survive.

## **Step 8: Correlation Analysis**
```python
corr = titanic.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
![Missing Values per Column](Titanic%20EDA/Titanic%20dataset%20images/Coding%20Samurai%209.png)
## Interpretation:
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

# Project 2

## **Natural Language Processing (NLP) - Sentiment Analysis**
## Introduction

## Objective:
The goal of this project is to perform **Sentiment Analysis** on social media data to understand public opinions, emotions, and attitudes expressed in online posts. By analyzing user-generated content such as tweets, we can automatically classify sentiments as **positive, negative, or neutral.**

This project demonstrates how Natural Language Processing (NLP) techniques can be applied to real-world text data to extract meaningful insights from human language — a vital skill in today’s data-driven and socially connected world.

## Approach:
We will collect social media data (e.g., from Twitter) and preprocess the text by cleaning noise such as emojis, hashtags, mentions, and URLs. Then, using NLP tools like **NLTK, TextBlob, or VADER,** we’ll evaluate the polarity and subjectivity of each post to determine its sentiment category.

## Learning Outcome:
By completing this project, we gain hands-on experience with:

- **Text preprocessing (tokenization, stopword removal, lemmatization)**

- **Sentiment classification using NLP libraries**

- **Data visualization to interpret overall sentiment trends**

Ultimately, this project bridges data analytics with language understanding — transforming unstructured text into actionable insights for decision-making in business, marketing, or social research.
- Preprocess text data (tokenization, removing
stopwords, and stemming/lemmatization).
- Use nltk or TextBlob for sentiment analysis.
- Visualize the sentiment distribution and word
frequencies using word clouds.
# Step 1: Import Libraries & Load Data
```python
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download punkt_tab resource

from google.colab import files
uploaded = files.upload()
df = pd.read_csv("3) Sentiment dataset.csv")

print(df.head())
```
## Interpretation:
We first load all required libraries for NLP, visualization, and sentiment scoring. Then, we import the Sentiment dataset and preview it. This ensures the data is correctly loaded and ready for processing.
# Step 2: Preprocess Text Data
```python
# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # Tokenize into words
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Preview cleaned text
print(df[['Text', 'cleaned_text']].head())
```
## Interpretation:
- The preprocessing pipeline ensures our text is standardized by:

- Removing numbers & special characters.

- Tokenizing words (splitting into individual terms).

- Removing stopwords (common words like “the”, “is”).

- Applying stemming (reducing words like running → run).

This prepares the text for accurate sentiment analysis.
# Step 3: Sentiment Classification
```python
# Sentiment classification function
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # -1 to +1
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply classification
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Preview results
print(df[['cleaned_text','sentiment']].head())
```
## Interpretation:

TextBlob assigns a polarity score to each text:

- greater than 0 → Positive sentiment

- greater than 0 → Negative sentiment

- = 0 → Neutral sentiment

This transforms raw text into structured sentiment categories.
# Step 4: Sentiment Distribution
```python
# Plot distribution
df['sentiment'].value_counts().plot(kind='bar', color=['gray','green','red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Reviews")
plt.show()
```
![Missing Values per Column](Sentiment%20Analysis/Sentiment%20Analysis%20Images/Coding%20Samurai%20NLP.png)
## Interpretation:
- This bar chart shows the balance of sentiments in the dataset.

- If one sentiment dominates, it may affect downstream NLP models.

- If balanced, it indicates a good dataset for training classifiers.
# Step 5: Word Cloud Visualizations
## Positive Reviews
```python
positive_text = " ".join(df[df['sentiment']=="Positive"]['cleaned_text'])
wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Positive Reviews", fontsize=14)
plt.show()
```
![Missing Values per Column](Sentiment%20Analysis/Sentiment%20Analysis%20Images/Coding%20Samurai%20NLP%202.png)
## Negative Reviews
```python
negative_text = " ".join(df[df['sentiment']=="Negative"]['cleaned_text'])
wordcloud_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Negative Reviews", fontsize=14)
plt.show()
```
![Missing Values per Column](Sentiment%20Analysis/Sentiment%20Analysis%20Images/Coding%20Samurai%20NLP%203.png)
## Neutral Reviews
```python
neutral_text = " ".join(df[df['sentiment']=="Neutral"]['cleaned_text'])
wordcloud_neu = WordCloud(width=800, height=400, background_color="gray", colormap="Blues").generate(neutral_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neu, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Neutral Reviews", fontsize=14)
plt.show()
```
![Missing Values per Column](Sentiment%20Analysis/Sentiment%20Analysis%20Images/Coding%20Samurai%20NLP%204.png)
## Interpretation:

- Positive Word Cloud: shows commonly used positive terms (e.g., great, love, amazing).

- Negative Word Cloud: highlights frequent negative words (e.g., bad, hate, worst).

- Neutral Word Cloud: captures neutral/common expressions (e.g., okay, average, fine).

This gives us deep insights into how people express emotions in text.

# ✅Final Professional Interpretation:

- Dataset successfully cleaned and preprocessed.

- Text classified into Positive, Negative, Neutral using TextBlob.

- Sentiment distribution visualized with bar charts.

- Word Clouds generated for all three sentiment categories to show key vocabulary patterns.

This pipeline is ready for reporting, dashboards, or as input for advanced NLP models like Logistic Regression, Naive Bayes, or Transformers.
# Author

**Duru Chukwuma**

📧 **chukwuduru588@gmail.com**

🔗 [LinkedIn](https://linkedin.com/in/chukwuma-duru)  
🔗 [Portfolio](https://www.datascienceportfol.io/chukwuduru588)

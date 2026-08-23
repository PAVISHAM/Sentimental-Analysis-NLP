# 🎬 Sentiment Analysis Using NLP

## 📌 Project Overview

🔗 Hugging Face Model Link: https://huggingface.co/pavisha-sentiment-analysis/SentimentalAnalysisModel/tree/main


🔗Live Demo Link :  https://drive.google.com/file/d/1ce2d6_fuMkbOVzO9HginxUTVSAZEL6-H/view

This project is a **Sentiment Analysis system** developed using **Natural Language Processing (NLP)** and Machine Learning.

The system analyzes movie reviews and classifies them into two sentiment categories:

* 😊 **Positive**
* 😞 **Negative**

The project uses the **IMDB Dataset of 50K Movie Reviews**. NLP techniques are applied to clean and preprocess the review text, followed by **TF-IDF Vectorization** to convert text into numerical features. A **Logistic Regression** model is then trained to classify the sentiment of reviews.



## 🛠️ Technologies Used

* **Python**
* **Natural Language Processing (NLP)**
* **Pandas**
* **Regular Expressions (re)**
* **Scikit-learn**
* **TF-IDF Vectorizer**
* **Logistic Regression**
* **Joblib**
* **KaggleHub**
* **Google Colab**

## 🔄 Project Workflow

```text
IMDB Movie Reviews Dataset
            ↓
       Data Loading
            ↓
    Sentiment Encoding
   Positive → 1
   Negative → 0
            ↓
      Text Cleaning
            ↓
   NLP Preprocessing
            ↓
     Train-Test Split
            ↓
     TF-IDF Vectorization
            ↓
     Logistic Regression
            ↓
      Model Prediction
            ↓
    Sentiment Evaluation
            ↓
 Positive / Negative

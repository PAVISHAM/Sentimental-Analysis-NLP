# 🎬 Sentiment Analysis Using NLP

## 📌 Project Overview


🔗Hugging Face Model Link :  https://huggingface.co/pavisha-sentiment-analysis/SentimentalAnalysisModel/tree/main


🔗Live Demo Link : https://drive.google.com/file/d/1ce2d6_fuMkbOVzO9HginxUTVSAZEL6-H/view


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
```

"

After preprocessing:
"i absolutely loved this movie it was amazing"
```

## 🔢 TF-IDF Vectorization

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the cleaned text into numerical feature vectors.

The project uses:

```python
TfidfVectorizer(
    stop_words='english',
    max_features=10000
)
```

The vectorizer uses the most important **10,000 features** while ignoring common English stop words.

## 🤖 Machine Learning Model

The project uses **Logistic Regression** for sentiment classification.

```python
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)
```

Logistic Regression is suitable for this task because it performs well for text classification problems when combined with TF-IDF features.

## 📈 Model Performance

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

The trained Logistic Regression model achieved:

### Accuracy: **88.97%**

```text
Accuracy = 0.8897
```

This means the model correctly classified approximately **89 out of every 100 movie reviews** in the test dataset.

## 🧪 Custom Review Prediction

The trained model was also tested with custom movie reviews.

### Example 1

```text
"I was on the edge of my seat! Absolute masterpiece with an incredible ending."
```

**Prediction:** 😊 Positive

### Example 2

```text
"Don't waste your money. The acting was wooden and the plot was incredibly boring."
```

**Prediction:** 😞 Negative

## 💾 Saving the Trained Model

The trained Logistic Regression model and TF-IDF vectorizer are saved using **Joblib**.

```python
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

The saved files are:

```text
sentiment_model.pkl
tfidf_vectorizer.pkl
-------

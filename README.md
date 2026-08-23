# 🎬 Sentiment Analysis Using NLP

## 📌 Project Overview

This project is a **Sentiment Analysis system** developed using **Natural Language Processing (NLP)** and Machine Learning.

The system analyzes movie reviews and classifies them into two sentiment categories:

* 😊 **Positive**
* 😞 **Negative**

The project uses the **IMDB Dataset of 50K Movie Reviews**. NLP techniques are applied to clean and preprocess the review text, followed by **TF-IDF Vectorization** to convert text into numerical features. A **Logistic Regression** model is then trained to classify the sentiment of reviews.

## 🎯 Objectives

* Analyze movie reviews using Natural Language Processing.
* Clean and preprocess raw text data.
* Convert textual data into numerical features using TF-IDF.
* Train a machine learning classification model.
* Evaluate the model using accuracy and classification metrics.
* Predict the sentiment of new/custom movie reviews.
* Save the trained model and TF-IDF vectorizer for future use.

## 📊 Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews**.

The dataset contains:

* **50,000 movie reviews**
* **25,000 reviews for training**
* **25,000 reviews for testing**
* Two sentiment classes:

  * Positive
  * Negative

The dataset is downloaded using `kagglehub`.

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

## 🧹 NLP Text Preprocessing

The raw movie reviews are cleaned before training the model.

The preprocessing steps include:

1. Removing HTML `<br>` tags.
2. Converting text to lowercase.
3. Removing special characters and numbers.
4. Removing extra whitespace.
5. Preparing the cleaned text for feature extraction.

Example:

```text
Original:
"I absolutely LOVED this movie! <br /> It was amazing."

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
```

These files can be loaded later to make predictions without retraining the model.

## 📁 Project Structure

```text
Sentiment-Analysis/
│
├── sentiment_analysis.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── requirements.txt
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd Sentiment-Analysis
```

### 3. Install Required Libraries

```bash
pip install pandas scikit-learn kagglehub joblib
```

### 4. Run the Notebook

Open the notebook in **Google Colab** or **Jupyter Notebook** and execute the cells.

## 📦 Required Libraries

```text
pandas
scikit-learn
kagglehub
joblib
```

## 🌍 Applications

Sentiment Analysis can be applied to:

* 🎬 Movie review analysis
* ⭐ Product review analysis
* 💬 Customer feedback analysis
* 📱 Social media sentiment analysis
* 🛍️ E-commerce review analysis
* 📊 Opinion mining

## 🔮 Future Enhancements

* Create a web interface for real-time sentiment prediction.
* Deploy the model using **Streamlit** or **Hugging Face Spaces**.
* Experiment with advanced NLP models.
* Improve text preprocessing techniques.
* Compare Logistic Regression with other classification algorithms.
* Implement deep learning models such as LSTM or Transformer-based models.

## 👩‍💻 Author

**Pavisha M**

BE Computer Science and Engineering
University College of Engineering Nagercoil

## ⭐ Conclusion

This project demonstrates how **Natural Language Processing and Machine Learning** can be combined to perform sentiment classification on movie reviews.

By using **text preprocessing, TF-IDF feature extraction, and Logistic Regression**, the system achieved an accuracy of **88.97%** on the test dataset and successfully predicted the sentiment of custom movie reviews.

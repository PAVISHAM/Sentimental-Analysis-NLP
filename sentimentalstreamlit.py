import streamlit as st
import re
import joblib

# -------------------------------
# Load Model and Vectorizer
# -------------------------------

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# -------------------------------
# Text Cleaning Function
# -------------------------------

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e1b4b 50%,
        #312e81 100%
    );
    color: white;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Main Title */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #a78bfa,
        #f472b6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 5px;
}


/* Subtitle */

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}


/* Information Card */

.info-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
}


/* Review Label */

.review-label {
    color: #e2e8f0;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
}


/* Text Area */

textarea {
    background-color: white !important;
    color: black !important;
    border: 2px solid #ec4899 !important;
    border-radius: 15px !important;
    font-size: 16px !important;
}

textarea::placeholder {
    color: #64748b !important;
}


/* Predict Button */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 13px;
    font-size: 18px;
    font-weight: 700;
    color: white;

    background: linear-gradient(
        90deg,
        #7c3aed,
        #ec4899
    );

    transition: 0.3s;

    box-shadow:
        0px 5px 20px
        rgba(139, 92, 246, 0.35);
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0px 8px 25px
        rgba(236, 72, 153, 0.45);
}


/* Footer */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# Header
# -------------------------------

st.markdown(
    '<div class="main-title">'
    '🎬 Movie Review Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered movie review analysis using Machine Learning & TF-IDF'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------
# Information Card
# -------------------------------

st.markdown("""
<div class="info-card">

<h3>🤖 How does it work?</h3>

<p>
This application analyzes your movie review
and predicts its sentiment.
</p>

<p>
😊 <b>Positive</b>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
😞 <b>Negative</b>
</p>

<p>
The prediction is performed using a Machine Learning
model trained on the IMDb movie review dataset.
</p>

</div>
""", unsafe_allow_html=True)


# -------------------------------
# User Input
# -------------------------------

st.markdown(
    '<div class="review-label">'
    '📝 Enter Your Movie Review'
    '</div>',
    unsafe_allow_html=True
)

review = st.text_area(
    "",
    height=180,
    placeholder=(
        "Example: This movie was absolutely amazing! "
        "The story, acting and music were fantastic..."
    )
)


# -------------------------------
# Prediction
# -------------------------------

if st.button("🔮 Predict Sentiment"):

    # Check empty input

    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review first.")

    else:

        # Clean review

        cleaned = clean_text(review)

        # Convert review to TF-IDF

        vector = vectorizer.transform([cleaned])

        # Predict sentiment

        prediction = model.predict(vector)[0]

        # Prediction probability

        probability = model.predict_proba(vector)


        # -------------------------------
        # Positive Prediction
        # -------------------------------

        if prediction == 1:

            confidence = float(probability[0][1])

            st.write("😊 Positive")
            st.write(
                f"Confidence: {confidence * 100:.2f}%"
            )


        # -------------------------------
        # Negative Prediction
        # -------------------------------

        else:

            confidence = float(probability[0][0])

            st.write("😞 Negative")
            st.write(
                f"Confidence: {confidence * 100:.2f}%"
            )


# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🎬 Movie Review Sentiment Analysis'
    '&nbsp; | &nbsp;'
    'Powered by Streamlit • Scikit-Learn • TF-IDF'
    '</div>',
    unsafe_allow_html=True
)
import streamlit as st
from transformers import pipeline

st.title("Simple Calculator & Sentiment Analyzer")

# --- Calculator Section ---
st.header("Calculator")

# Function to perform calculation
def calculate(num1, num2, operation):
    try:
        if operation == "Add":
            return num1 + num2
        elif operation == "Subtract":
            return num1 - num2
        elif operation == "Multiply":
            return num1 * num2
        elif operation == "Divide":
            if num2 == 0:
                return "Error: Division by zero"
            else:
                return num1 / num2
        else:
            return "Invalid operation"
    except Exception as e:
        return f"Error: {e}"

# Get calculator input
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("Enter first number", value=0.0, format="%f", key="num1")
with col2:
    num2 = st.number_input("Enter second number", value=0.0, format="%f", key="num2")

operation = st.selectbox(
    "Choose an operation",
    ("Add", "Subtract", "Multiply", "Divide"),
    key="operation"
)

# Calculate and display result
if st.button("Calculate", key="calculate_button"):
    result = calculate(num1, num2, operation)
    st.success(f"Calculator Result: {result}")

st.divider()

# --- Sentiment Analysis Section ---
st.header("Sentiment Analyzer")

# Load the sentiment analysis pipeline (cache it for performance)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# Get text input
text_input = st.text_area("Enter text for sentiment analysis:", key="sentiment_text")

# Analyze sentiment and display result
if st.button("Analyze Sentiment", key="analyze_button"):
    if text_input:
        try:
            result = sentiment_analyzer(text_input)
            sentiment = result[0]['label']
            score = result[0]['score']
            st.success(f"Sentiment: {sentiment} (Score: {score:.4f})")
        except Exception as e:
            st.error(f"Error during analysis: {e}")
    else:
        st.warning("Please enter some text to analyze.")



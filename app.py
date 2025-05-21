
import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# --------- Helper functions ---------

def clean_answer_text(answer):
    # Remove leading "A:", "Answer:", "Q:", or numbering like "1.", "2)" etc.
    answer = re.sub(r'^(A:|Answer:|Q:|\d+[\.\)])\s*', '', answer, flags=re.IGNORECASE)
    # Remove extra spaces/newlines
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()

def extract_qa_pairs(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    qa_pairs = []
    current_q = None
    current_a = []

    for line in lines:
        if line.endswith('?'):
            if current_q and current_a:
                answer_text = " ".join(current_a).strip()
                answer_text = clean_answer_text(answer_text)
                qa_pairs.append((current_q, answer_text))
            current_q = line
            current_a = []
        else:
            if current_q:
                current_a.append(line)
    # Add last Q&A pair
    if current_q and current_a:
        answer_text = " ".join(current_a).strip()
        answer_text = clean_answer_text(answer_text)
        qa_pairs.append((current_q, answer_text))
    return qa_pairs

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess(text):
    # Lowercase and remove punctuation for better similarity matching
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def get_best_match(user_query, faq_data):
    corpus = [preprocess(q + " " + a) for q, a in faq_data]
    processed_query = preprocess(user_query)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus + [processed_query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_idx = similarity.argmax()
    return faq_data[best_idx][1]  # Return only the answer part

# --------- Streamlit UI ---------

st.set_page_config(page_title="Deepfake Detection FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("📘 Deepfake Detection FAQ Chatbot")
st.write("Ask questions related to Deepfake Detection and get answers from the fixed FAQ PDF.")

# Load PDF once on startup (make sure deepfake_faq.pdf is in same folder)
try:
    with open("deepfake_faq.pdf", "rb") as f:
        raw_text = extract_text_from_pdf(f)
        faq_data = extract_qa_pairs(raw_text)
except FileNotFoundError:
    st.error("FAQ PDF file (deepfake_faq.pdf) not found in the app folder.")
    st.stop()

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Searching for the best answer..."):
        answer = get_best_match(user_input, faq_data)
    st.markdown("### 🤖 Answer:")
    st.write(answer)

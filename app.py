
import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# --------- Theme Toggle ---------
dark_mode = st.toggle("🌙 Dark Mode", value=False)

# Apply theme CSS
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0e1117;
            color: white;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #262730;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

# --------- Helper functions ---------

def clean_answer_text(answer):
    answer = re.sub(r'^(A:|Answer:|Q:|\d+[\.\)])\s*', '', answer, flags=re.IGNORECASE)
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
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def get_best_match(user_query, faq_data):
    corpus = [preprocess(q + " " + a) for q, a in faq_data]
    processed_query = preprocess(user_query)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus + [processed_query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_idx = similarity.argmax()
    return faq_data[best_idx][1]

# --------- UI ---------

st.title("Deepfake Detection FAQ Chatbot")
st.write("Hi!! Ask questions related to Deepfake Detection and get answers.")

try:
    with open("deepfake_faq.pdf", "rb") as f:
        raw_text = extract_text_from_pdf(f)
        faq_data = extract_qa_pairs(raw_text)
except FileNotFoundError:
    st.error("FAQ PDF file (deepfake_faq.pdf) not found.")
    st.stop()

# Use a form to allow Enter to submit
with st.form(key="qa_form"):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Get Answer")

if submitted and user_input:
    with st.spinner("Searching for the best answer..."):
        answer = get_best_match(user_input, faq_data)
    st.markdown("### 🤖 Answer:")
    st.write(answer)

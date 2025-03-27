from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os


# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of app.py
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_nb_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

with open(MODEL_PATH, 'rb') as f:
    best_nb_model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


# Define request model
class Review(BaseModel):
    review: str

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

@app.post("/predict")
async def predict(review: Review):
    cleaned_review = preprocess_text(review.review)

    # Handle empty reviews
    if not cleaned_review.strip():
        return {"sentiment": "neutral"}

    review_tfidf = tfidf_vectorizer.transform([cleaned_review])
    prediction = best_nb_model.predict(review_tfidf)[0]
    return {"sentiment": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

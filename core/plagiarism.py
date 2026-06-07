import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Download minimal NLTK requirements locally
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

class PlagiarismDetector:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
            
        return " ".join([t for t in tokens if t not in self.stop_words])

    def get_word_stats(self, text1: str, text2: str) -> dict:
        """Extracts word counts and common matching words."""
        clean1 = self.clean_text(text1).split()
        clean2 = self.clean_text(text2).split()
        
        common_words = set(clean1).intersection(set(clean2))
        
        return {
            "word_count_A": len(clean1),
            "word_count_B": len(clean2),
            "common_words": list(common_words)[:10]  # top 10 simple matches
        }

    def detect_plagiarism(self, doc1: str, doc2: str) -> dict:
        """Detects plagiarism percentage and levels between two texts."""
        if not doc1.strip() or not doc2.strip():
            return {"score": 0, "level": "Low Plagiarism", "color": "Green"}

        clean1 = self.clean_text(doc1)
        clean2 = self.clean_text(doc2)

        if not clean1 or not clean2:
            return {"score": 0, "level": "Low Plagiarism", "color": "Green"}

        # Cosine Similarity via TF-IDF
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([clean1, clean2])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            score = 0.0

        score_pct = int(score * 100)
        
        # Categorize
        if score_pct > 60:
            level = "High Plagiarism"
            color = "Red"
        elif score_pct > 30:
            level = "Medium Plagiarism"
            color = "Orange"
        else:
            level = "Low Plagiarism"
            color = "Green"

        return {
            "score": score_pct,
            "level": level,
            "color": color
        }

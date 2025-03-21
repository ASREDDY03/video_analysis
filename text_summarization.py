import spacy
from langdetect import detect
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

def detect_language(text):
    """Detects the language of a given text."""
    try:
        return detect(text)  # Returns language code (e.g., 'en', 'fr', 'es')
    except:
        return "unknown"  # If detection fails, return 'unknown'

class CustomTokenizer:
    """Custom tokenizer for Sumy that uses SpaCy instead of NLTK."""
    def __init__(self, language):
        self.language = language

    def to_sentences(self, text):
        """Tokenizes text into sentences using SpaCy."""
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    def to_words(self, text):
        """Tokenizes text into words using SpaCy."""
        doc = nlp(text)
        return [token.text for token in doc if not token.is_punct]  # Exclude punctuation

def summarize_text(text):
    """Summarizes text using Sumy with a custom SpaCy tokenizer."""
    if not text or text.strip() == "":
        return "No speech detected."

    # Use CustomTokenizer to override Sumy's default NLTK tokenizer
    parser = PlaintextParser.from_string(text, CustomTokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Extract top 3 sentences
    
    return " ".join([str(sentence) for sentence in summary])

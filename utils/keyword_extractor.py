import re
from nltk.corpus import stopwords

class KeywordExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_keywords(self, text, num_keywords=5):
        words = re.findall(r'\b[a-z]+\b', text.lower())

        keywords = [
            w for w in words
            if w not in self.stop_words and len(w) > 2
        ]

        return list(dict.fromkeys(keywords))[:num_keywords]
import nltk
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class FeatureExtractor:
    def __init__(self, method='bm25', sbert_model_name='paraphrase-multilingual-MiniLM-L12-v2', max_features=1000):
        self.method = method
        self.batch_size = 16 

        if method == 'bm25':
            self.tokenizer = nltk.word_tokenize
            self.bm25 = None  
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
        elif method == 'sbert':
            self.sbert_model = SentenceTransformer(sbert_model_name)
        else:
            raise ValueError("Method not supported. Choose 'bm25', 'tfidf', or 'sbert'.")

    def fit_transform(self, texts):
        if self.method == 'bm25':
            return self._bm25_vectorize(texts)
        elif self.method == 'tfidf':
            return self.vectorizer.fit_transform(texts)
        elif self.method == 'sbert':
            return self._sbert_vectorize(texts)

    def transform(self, texts):
        if self.method == 'bm25':
            return self._bm25_transform(texts)
        elif self.method == 'tfidf':
            return self.vectorizer.transform(texts)
        elif self.method == 'sbert':
            return self._sbert_vectorize(texts)

    def _bm25_vectorize(self, texts):
        tokenized_texts = [self.tokenizer(text.lower()) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        return tokenized_texts
 
    def _bm25_transform(self, texts):
        tokenized_query = [self.tokenizer(text.lower()) for text in texts]
        return [self.bm25.get_scores(query) for query in tokenized_query]

    def _sbert_vectorize(self, texts):
        return self.sbert_model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)

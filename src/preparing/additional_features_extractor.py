import re
import spacy
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


nlp = spacy.load("en_core_web_sm")


sentiment_analyzer = SentimentIntensityAnalyzer()

class AdditionalFeatureExtractor:
    def __init__(self, 
                 use_ner=True, 
                 use_sentiment=True, 
                 use_emoji_embeddings=False, 
                 emoji_extractor=None, 
                 use_url=True, 
                 use_hashtags=True, 
                 use_keywords=True):  
        self.use_ner = use_ner
        self.use_sentiment = use_sentiment
        self.use_emoji_embeddings = use_emoji_embeddings
        self.emoji_extractor = emoji_extractor
        self.use_url = use_url
        self.use_hashtags = use_hashtags
        self.use_keywords = use_keywords

    def extract_ner(self, text):
        doc = nlp(text)
        # Extract NER types as a multi-hot vector
        ner_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME']
        multi_hot_vector = [0] * len(ner_types)
        for ent in doc.ents:
            if ent.label_ in ner_types:
                multi_hot_vector[ner_types.index(ent.label_)] = 1
        return multi_hot_vector

    def extract_sentiment(self, text):
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        return sentiment_scores['compound']

    def extract_emoji_embeddings(self, text):
        if not self.emoji_extractor:
            return []
        emoji_embeddings = []
        emojis_in_text = emoji.emoji_list(text)
        for emoji_info in emojis_in_text:
            emoji_char = emoji_info['emoji']
            embedding = self.emoji_extractor.get_emoji_embedding(emoji_char)
            if embedding is not None:
                emoji_embeddings.append(embedding)
        return emoji_embeddings

    def extract_url_features(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls = url_pattern.findall(text)
        return len(urls)  

    def extract_hashtags(self, text):
        hashtags = re.findall(r'#\w+', text)
        return len(hashtags)  

    def extract_keywords(self, text):
        keywords = []
        # Extract URL keywords
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls = url_pattern.findall(text)
        for url in urls:
            processed_url = re.sub(r'https?://(www\.)?', '', url)
            processed_url = re.sub(r'\W+', ' ', processed_url)
            keywords.extend(processed_url.split())
        # Extract Hashtags
        hashtags = re.findall(r'#\w+', text)
        keywords.extend([hashtag.strip('#') for hashtag in hashtags])
        return keywords

    def extract_features(self, text):
        features = {}

        if self.use_ner:
            features['ner_vector'] = self.extract_ner(text)
        
        if self.use_sentiment:
            features['sentiment'] = self.extract_sentiment(text)
        
        if self.use_emoji_embeddings:
            features['emoji_embeddings'] = self.extract_emoji_embeddings(text)
        
        if self.use_url:
            features['url_count'] = self.extract_url_features(text)
        
        if self.use_hashtags:
            features['hashtag_count'] = self.extract_hashtags(text)

        if self.use_keywords:
            features['keywords'] = self.extract_keywords(text)

        return features

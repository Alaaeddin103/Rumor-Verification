import re
import string
import emoji

class Preprocessor:
    def __init__(self):
        pass    

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+|https?:\S+')
        return url_pattern.sub(r'', text)
  
    def remove_special_characters(self, text):  
        # remove punctuation and the specific quotation marks
        special_characters = string.punctuation + "“”‘’"
        return text.translate(str.maketrans('', '', special_characters))

    def remove_noise_words(self, text):
        # remove RT
        text = re.sub(r"\bRT\b\s*", "", text)
        return text

    def remove_emojis(self, text):
        
        return emoji.replace_emoji(text, replace='')

    def remove_hashtags(self, text):
        hashtag_pattern = re.compile(r'#\w+')
        return hashtag_pattern.sub(r'', text)

    def remove_mentions(self, text):
        mention_pattern = re.compile(r'@\w+')
        return mention_pattern.sub(r'', text)
    
    def remove_white_spaces(self, text):
        # Clean up extra white spaces
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_text(self, text):
        text = self.remove_urls(text)
        text = self.remove_hashtags(text)
        text = self.remove_mentions(text)
        text = self.remove_emojis(text)
        text = self.remove_noise_words(text)
        text = self.remove_special_characters(text)
        text = self.remove_white_spaces(text) 

        return text

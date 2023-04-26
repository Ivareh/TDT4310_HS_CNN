import pandas as pd
import re
import numpy as np
from transformers import BertTokenizer

def preprocess_tweets(tweets):
    # Convert tweet to lower case
    for tweet in tweets:
        # Replace URLs with <url> token
        tweet = re.sub(r'http\S+', '<url>', tweet)
        # Replace usernames with <user> token
        tweet = re.sub(r'@\S+', '<user>', tweet)
        # Replace hashtags with their textual content
        tweet = re.sub(r'#(\S+)', lambda match: re.sub(r'(?<!^)(?=[A-Z])', ' ', match.group(1)), tweet)
        # Replace hashtags with <hashtag> token
        tweet = re.sub(r'#\S+', '<hashtag>', tweet)
        # Replace hashtags with <hashtag> token
        tweet = re.sub(r'#\B', '<hashtag>', tweet)
        # Replace numbers with <number> token
        tweet = re.sub(r'\b\d+\b', '<number>', tweet)
        # Replace elongated words with their base form
        tweet = re.sub(r'(\w)\1{2,}', r'\1', tweet)
        # Replace common emoticons with <emoticon> token
        emoticons = [(':\')', '<emoticon>'), (':-)', '<emoticon>'), (':)', '<emoticon>'), (':-(', '<emoticon>'), (':(', '<emoticon>'), (':P', '<emoticon>'), (':p', '<emoticon>'), (':D', '<emoticon>'), (':d', '<emoticon>'), (':O', '<emoticon>'), (':o', '<emoticon>'), (':/', '<emoticon>'), (':|', '<emoticon>'), (';)', '<emoticon>'), (':*', '<emoticon>'), (':$', '<emoticon>'), ('8)', '<emoticon>'), ('8-)', '<emoticon>'), (':\'(', '<emoticon>')]
        for emoticon, token in emoticons:
            tweet = tweet.replace(emoticon, token)
        # Remove punctuation marks and unknown unicodes
        tweet = re.sub(r'[^\w\s<>]|_[^\w\s<>]|(?<=_)\d+|[^\x00-\x7F]+', '', tweet)
        # Remove extra delimiting characters
        tweet = re.sub(r' +', ' ', tweet)
        tweet = tweet.strip()
    return tweets

def read_dataset():
    # Read the CSV file containing the tweets
    data = pd.read_csv('./data/labeled_data.csv')
    data.drop(['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
    return data["tweet"].tolist(), data["class"]

def preprocess_dataset(data, labels):
    input_ids = []
    attention_masks = []
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for sentence in data:
        bert_inp = bert_tokenizer.__call__(
            sentence, 
            max_length=50,
            padding='max_length', 
            pad_to_max_length=True,
            truncation=True, 
            return_token_type_ids=False)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])
    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels)
    return input_ids, attention_masks, labels


def load_process_data():
    data, labels = read_dataset()
    input_ids, attention_masks, labels = preprocess_dataset(preprocess_tweets(data), labels)
    return input_ids, attention_masks, labels



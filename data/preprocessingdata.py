import pandas as pd
import re

def preprocess_tweet(tweet):
    # Convert tweet to lower case
    tweet = tweet.lower()
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
    return tweet

# Read the CSV file containing the tweets
df = pd.read_csv('./data/labeled_data.csv')

# Preprocess the tweets
df['tweet'] = df['tweet'].apply(preprocess_tweet)

# Save the preprocessed tweets to a new CSV file
df.to_csv('./data/preprocessed/preprocessed_tweets.csv', index=False)

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tweet_cleaner(tweet):
    # Convert to string and lowercase first
    tweet = str(tweet).lower()
    
    # Remove URLs 
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)
    
    # Remove mentions (@username)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Handle hashtags - remove the # but keep the word
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Remove numbers and special characters, but keep important punctuation
    tweet = re.sub(r'[^a-zA-Z\s\']', ' ', tweet)
    
    # Remove single quotes if they're not part of contractions
    tweet = re.sub(r"(?<!\w)'|'(?!\w)", '', tweet)
    
    # Remove emojis and non-ASCII characters
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')

    # Expand contractions
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"can't", "can not", tweet)
    
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    
    # Remove stopwords and short words
    words = tweet.split()
    cleaned = [word for word in words if word not in stop_words and len(word) >= 3]
    
    # Handle empty tweets
    if not cleaned:
        return ""
    
    return ' '.join(cleaned)

##Tags the words in the tweets
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:          
        return(None)

##Lemmatizes the words in tweets and returns the cleaned and lemmatized tweet
def lemmatize_tweet(tweet):
    #tokenize the tweet and find the POS tag for each token
    tweet = tweet_cleaner(tweet) #tweet_cleaner() will be the function you will write
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_tweet.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return(" ".join(lemmatized_tweet))
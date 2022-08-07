import nltk  # Python library for NLP
from nltk.corpus import twitter_samples  # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt  # library for visualization
import random  # pseudo-random number generator
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings



def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

if __name__ == "__main__":
    # downloads sample twitter dataset.
    # nltk.download('twitter_samples')
    # download the stopwords from NLTK
    nltk.download('stopwords')

    # select the set of positive and negative tweets
    # in this we now we have list of positive and negative tweets, each tweet is a string
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    print('Number of positive tweets: ', len(all_positive_tweets))
    print('Number of negative tweets: ', len(all_negative_tweets))

    print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
    print('The type of a tweet entry is: ', type(all_negative_tweets[0]))

    # print some tweets
    for tweet in all_positive_tweets[0:5]:
        print(tweet)

    # choose one random positive tweet and one negative
    # print positive in green
    print('\033[92m' + all_positive_tweets[random.randint(0, 5000)])

    # print negative in red
    print('\033[91m' + all_negative_tweets[random.randint(0, 5000)])

    # Our selected sample. Complex enough to exemplify each step
    tweet = all_positive_tweets[2177]
    print(tweet)

    print('\033[92m' + tweet)
    print('\033[94m')

    # remove old style retweet text "RT"
    tweet2 = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet2 = re.sub(r'#', '', tweet2)

    print(tweet2)

    print('\033[92m' + tweet2)
    print('\033[94m')


    # tokanisation (the tweet tokinser will resirve the smily faces and hashtags )
    # strip_handles will remove the @HolmesdaleCC (name of the one who tweet)
    # reduce_len: will delete the repeated latters alot like ("waaaaaaayyyyy") >> waaayyy
    # instantiate tokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(tweet2)

    print()
    print('Tokenized string:')
    print(tweet_tokens)



    #Import the english stop words list from NLTK
    stopwords_english = stopwords.words('english')

    print('Stop words\n')
    print(stopwords_english)

    print('\nPunctuation\n')
    print(string.punctuation)

    print()
    print('\033[92m')
    print(tweet_tokens)
    print('\033[94m')

    tweets_clean = []

    for word in tweet_tokens:  # Go through every word in your tokens list
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            tweets_clean.append(word)

    print('removed stop words and punctuation:')
    print(tweets_clean)

    print()
    print('\033[92m')
    print(tweets_clean)
    print('\033[94m')

    # Instantiate stemming class
    stemmer = PorterStemmer()

    # Create an empty list to store the stems
    tweets_stem = []

    for word in tweets_clean:
        stem_word = stemmer.stem(word)  # stemming word
        tweets_stem.append(stem_word)  # append to the list

    print('stemmed words:')
    print(tweets_stem)

    # choose the same tweet
    tweet = all_positive_tweets[2277]

    print()
    print('\033[92m')
    print(tweet)
    print('\033[94m')

    # call the imported function
    tweets_stem = process_tweet(tweet)  # Preprocess a given tweet

    print('preprocessed tweet:')
    print(tweets_stem)  # Print the result




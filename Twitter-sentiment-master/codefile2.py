import pandas as pd
import numpy as np
import nltk
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

#from sklearn.model_selection import train_test_split

path = '/home/sandynote/Desktop/Hackathon/JantaHack4/train_2kmZucJ.csv'#sample_submission_LnhVWA4.csv
data = pd.read_csv(path)

x_train = data[:5500]
x_test = data[5500:]

## prepossing on x_train dataset

# removing digits from tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] = ''.join([j for j in x_train['tweet'][i] if not j.isdigit()])
    
# removing punctuation
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] = ''.join([j for j in x_train['tweet'][i] if j not in string.punctuation])


# removing url/hyperlinks form tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] =  re.sub(r"http\S+", "", x_train['tweet'][i])
    
# removing # from tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] =  re.sub(r'#', "", x_train['tweet'][i])

# instantiate tokenizer class   
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
# tokenize tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] =  tokenizer.tokenize(x_train['tweet'][i])
    

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english') 

x_train['tweet1']= ""
for i in range(len(x_train['tweet'])):
    x_train['tweet1'][i] = []
    for word in x_train['tweet'][i]:
        if (word not in stopwords_english and word not in string.punctuation):
           x_train['tweet1'][i].append(word)


# Instantiate stemming class
stemmer = PorterStemmer() 
x_train['tweet2'] = ""
for i in range(len(x_train['tweet1'])):
    x_train['tweet2'][i] = []
    for word in x_train['tweet1'][i]:
        stem_word = stemmer.stem(word)  # stemming word
        x_train['tweet2'][i].append(stem_word)  # append to the list


## prepossing on x_test dataset
x_test.reset_index(inplace = True)
# removing digits from tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] = ''.join([j for j in x_test['tweet'][i] if not j.isdigit()])
    
# removing punctuation
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] = ''.join([j for j in x_test['tweet'][i] if j not in string.punctuation])


# removing url/hyperlinks form tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] =  re.sub(r"http\S+", "", x_test['tweet'][i])
    
# removing # from tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] =  re.sub(r'#', "", x_test['tweet'][i])

# instantiate tokenizer class   
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
# tokenize tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] =  tokenizer.tokenize(x_test['tweet'][i])
    

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english') 

x_test['tweet1']= ""
for i in range(len(x_test['tweet'])):
    x_test['tweet1'][i] = []
    for word in x_test['tweet'][i]:
        if (word not in stopwords_english and word not in string.punctuation):
           x_test['tweet1'][i].append(word)


# Instantiate stemming class
stemmer = PorterStemmer() 
x_test['tweet2'] = ""
for i in range(len(x_test['tweet1'])):
    x_test['tweet2'][i] = []
    for word in x_test['tweet1'][i]:
        stem_word = stemmer.stem(word)  # stemming word
        x_test['tweet2'][i].append(stem_word)  # append to the list

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    
    for y, tweet in zip(yslist, tweets):
        for word in tweet:
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
                
    return freqs

freqs_train = build_freqs(x_train['tweet2'], x_train['label'])
freqs_test = build_freqs(x_test['tweet2'], x_test['label'])


def lookup(freqs, word, label):
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n



def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    train_y = np.array(train_y).reshape((len(train_y), 1))

    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]

    # Calculate D, the number of documents
    D = len(train_x)

    # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
    D_pos = np.sum(train_y == 1)

    # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    D_neg = np.sum(train_y == 0)

    # Calculate logprior
    logprior =  np.log(D_pos) - np.log(D_neg)
    
    df = pd.DataFrame()

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)
        
        data1 = {"vocab":word,"p_w_pos":p_w_pos, "p_w_neg":p_w_neg, "loglikelihood":loglikelihood[word]}             # fitting the above data to the empty datafrme
        df = df.append(data1, ignore_index=True)


    return logprior, loglikelihood, df
#df_test = pd.DataFrame()
#data = {"category":label,"img_path":img_path[0], "img_vec":output_FC}             # fitting the above data to the empty datafrme
#      df_test = df_test.append(data, ignore_index=True) 



logprior_train, loglikelihood_train, df_train = train_naive_bayes(freqs_train, x_train['tweet2'], x_train['label'])
logprior_test, loglikelihood_test, df_test = train_naive_bayes(freqs_test, x_test['tweet2'], x_test['label'])

def naive_bayes_predict(tweet, logprior, loglikelihood):
    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in tweet:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0 
    
    test_y = np.array(test_y).reshape((len(test_y), 1))

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    #error = np.mean(np.abs(y_hats - test_y))
    accuracy = sum(sum(np.array(y_hats) == np.array(test_y.T)))/len(y_hats)

    # Accuracy is 1 minus the error
    #accuracy = 1 - error


    return accuracy

acc_train = test_naive_bayes(x_train['tweet2'], x_train['label'], logprior_train, loglikelihood_train)
acc_test = test_naive_bayes(x_test['tweet2'], x_test['label'], logprior_test, loglikelihood_test)

print(acc_train)
print(acc_test)


def get_ratio(freqs, word):
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)

    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)

    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    return pos_neg_ratio

def get_words_by_threshold(freqs, label, threshold):
    word_list = {}

    for key in freqs.keys():
        word, _ = key

        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio


        # otherwise, do not include this word in the list (do nothing)

    return word_list

zero = get_words_by_threshold(freqs_test, label=0, threshold=0.05)
one = get_words_by_threshold(freqs_test, label=1, threshold=0.05)




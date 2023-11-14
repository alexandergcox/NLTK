# Advanced Analytics for Organisational Impact


# - how to apply the Naive Bayes classifier
# - how to interpret and communicate your findings.
# 

# # 1. Prepare your workstation

# In[1]:


# Import the necessary library.
import nltk

# Download the existing movie reviews.
nltk.download('movie_reviews')


# # 

# # 2. Costruct a list of documents

# In[2]:


# Import the necessary libraries.
from nltk.corpus import movie_reviews
import random

# Construct a nested list of documents.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Reorganise the list randomly.
random.shuffle(documents)


# In[3]:


# Create a list of files with negative reviews.
negative_fileids = movie_reviews.fileids('neg')

# Create a list of files with positive reviews.
positive_fileids = movie_reviews.fileids('pos')

# Display the list lengths.
print(len(negative_fileids), len(positive_fileids))


# In[4]:


# View the output.
print(movie_reviews.raw(fileids=positive_fileids[0]))


# # 

# # 3. Define a feature extractor function

# In[5]:


# Create an object to contain the frequency distribution.
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# Create a list that contains the first 2,000 words.
word_features = list(all_words)[:2000]

# Define a function to check whether each word is in the set of features.
def document_features(document): 
    # Create a set of document words.
    document_words = set(document)
    # Create an empty dictionary of features.
    features = {}
    # Populate the dictionary.
    for word in word_features:
       # Specify whether each feature exists in the set of document words. 
       features['contains({})'.format(word)] = (word in document_words)
    # Return the completed dictionary.
    return features


# In[6]:


# Generate a dictionary for the first review.
test_result = document_features(documents[0][0])

for key in test_result:
    print(key, ' : ', test_result[key])


# # 

# # 4. Train the classifier

# In[7]:


# Create a list of feature sets based on the documents list.
featuresets = [(document_features(d), c) for (d, c) in documents]

# Assign items to the training and test sets.
# Note the first and last 100 only.
train_set, test_set = featuresets[100:], featuresets[:100]

# Create a classifier object trained on items from the training set.
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Display the accuracy score in comparison with the test set.
print(nltk.classify.accuracy(classifier, test_set))


# > **Note:** Your output may differ from the demonstration output due to the `random.shuffle()` method applied to the data and possible updates to the `movie_reviews` corpus.

# # 

# # 5. Interpret the results

# In[8]:


# You can change the number of outputs by increasing or decreasing the number in the brackets.
classifier.show_most_informative_features(10)


# > **Note:** Your output may differ from the demonstration output due to the `random.shuffle()` method applied to the data and possible updates to the `movie_reviews` corpus.

# # 

# # 6. Conclusion(s)

# >This quick demonstration shows that the Naive Bayes sentiment classifier is relatively easy to interpret. This transparency is one of the core advantages of the model. The main limitation is the assumption of independent predictors. Realistically, mostly all text is contextual and can only be understood in relation to the other text surrounding it. So remember to interpret the results with caution.

# # 

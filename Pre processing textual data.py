#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install NLTK.
get_ipython().system('pip install nltk')


# In[3]:


# Import the library.
import nltk

# Install the required tokenisation model.
nltk.download('punkt')

# Install the required tokenisation function.
from nltk.tokenize import sent_tokenize


# In[4]:


# Assign the raw text data to a variable.
text = """We took this ball to the beach and after close to 2 hours to pump it up, we pushed it around for about 10 fun filled minutes. That was when the wind picked it up and sent it huddling down the beach at about 40 knots. It destroyed everything in its path. Children screamed in terror at the giant inflatable monster that crushed their sand castles. Grown men were knocked down trying to save their families. The faster we chased it, the faster it rolled. It was like it was mocking us. Eventually, we had to stop running after it because its path of injury and destruction was going to cost us a fortune in legal fees. Rumor has it that it can still be seen stalking innocent families on the Florida panhandle. We lost it in South Carolina, so there is something to be said about its durability."""

# Tokenise the text data.
tokenised_sentence = sent_tokenize(text)

# Check the result.
print(tokenised_sentence)


# In[5]:


# Import the function.
from nltk.tokenize import word_tokenize

# Tokenise the text data.
tokenised_word = word_tokenize(text)

# Check the result.
print(tokenised_word)


# In[6]:


# Import the class.
from nltk.probability import FreqDist

# Create a frequency distribution object.
freq_dist_of_words = FreqDist(tokenised_word)

# Show the five most common elements in the data set.
freq_dist_of_words.most_common(5)


# In[7]:


# Import the package.
import matplotlib.pyplot as plt

# Define the figure and axes.
fig, ax = plt.subplots(dpi=100)
fig.set_size_inches(12, 12)

# Plot the data set.
freq_dist_of_words.plot(30, cumulative=False)

# Display the result.
plt.show()


# In[8]:


# Download the stopwords.
nltk.download('stopwords')

# Import the package.
from nltk.corpus import stopwords

# Create a set of English stopwords.
stop_words = set(stopwords.words('english'))

# Display the set.
print(stop_words)


# In[9]:


# Create an empty list for the filtered words.
filtered_text = []

# Tokenise the text data.
# Create a tokenised word list.
tokenised_word = nltk.word_tokenize(text)

# Filter the tokenised words.
for each_word in tokenised_word:
    if each_word not in stop_words:
        filtered_text.append(each_word)
        
# Display the filtered list.      
print("Tokenised list without stopwords: {}".format(filtered_text))


# In[10]:


# Import the necessary class.
from nltk.stem.snowball import SnowballStemmer

# Download the resource.
nltk.download('wordnet')

# Create a stemming object.
snow_stem = SnowballStemmer(language='english')

# Create a list of test words.
words = ['easily', 'durability', 'longest', 'wishing', 'worthwhile',
         'fantasizing', 'off-putting']

# Apply the stemming process to each word.
for word in words:
    print(word + "--->" + snow_stem.stem(word))


# In[11]:


# Import the necessary class.
from nltk.stem.wordnet import WordNetLemmatizer

# Download the lemma corpus.
nltk.download('omw-1.4')

# Create an instance of the class.
lemmatiser = WordNetLemmatizer()

# Create an empty output list.
lemmatised_words_list = []

# Define a text string to test.
text_2 = "I love it when he purchases items that the kids and grandkids can't wait to try out. it's a lot of fun but he easily and accidentally bowls the toddlers over with them, so be careful."

# Tokenise the string.
tokenised_word = word_tokenize(text_2)

# Apply lemmatisation to each tokenised word.
for each_word in tokenised_word:
    lem_word = lemmatiser.lemmatize(each_word)
    lemmatised_words_list.append(lem_word)

# Display the output list.
print("Lemmatised Words list {}".format(lemmatised_words_list))


# In[12]:


# Import the necessary module.
import re

# Define some text. 
text_3 = "Perfect! Buying a second. Using it to make a hot air balloon for an escape room adventure. Event in Oct. will share photos."

# Filter out the specified punctuation.
no_punct = re.sub(r"[\.\?\!\,\:\;\"]", '', text_3)

# Display the filtered text.
print(no_punct)


# In[ ]:





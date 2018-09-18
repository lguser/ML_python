
# coding: utf-8

# # Naive Bayes (the easy way)

# We'll cheat by using sklearn.naive_bayes to train a spam classifier! Most of the code is just loading our training data into a pandas DataFrame that we can play with:

# In[1]:


import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.externals import joblib


# In[7]:


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})


# In[10]:


import os
os.getcwd()


# In[8]:


data = data.append(dataFrameFromDirectory('C:\AI_ML\ML_python\models\email_spam\emails\spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:\AI_ML\ML_python\models\email_spam\emails\ham', 'ham'))


# Let's have a look at that DataFrame:

# In[5]:


data.head(5)


# Now we will use a CountVectorizer to split up each message into its list of words, and throw that into a MultinomialNB classifier. Call fit() and we've got a trained spam filter ready to go! It's just that easy.

# In[16]:


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
counts


# Let's try it out:

# In[19]:


mymail = ['Free Demo will be there in the week end please regiser your self',"Free demo for free loan"]
example_counts = vectorizer.transform(mymail)
predictions = classifier.predict(example_counts)
predictions


# ## Activity

# Our data set is small, so our spam classifier isn't actually very good. Try running some different test emails through it and see if you get the results you expect.
# 
# If you really want to challenge yourself, try applying train/test to this spam classifier - see how well it can predict some subset of the ham and spam emails.

# In[9]:


joblib.dump(classifier,'email_model.pkl')


# In[10]:


email = joblib.load('email_model.pkl')
email


# In[16]:


examples = ['techinical details',"project discussion"]
example_counts = vectorizer.transform(examples)
email.predict(example_counts)


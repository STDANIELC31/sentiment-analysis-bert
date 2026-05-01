# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
from wordcloud import WordCloud
# %matplotlib inline
nltk.download('punkt')
nltk.download('wordnet')

# Torch ML libraries
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Misc.
import warnings
warnings.filterwarnings('ignore')

# Commented out IPython magic to ensure Python compatibility.
# Set intial variables and constants
# %config InlineBackend.figure_format='retina'

# Random seed for reproducibilty
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv('')
df.shape

df.head()

# separating the columns
spotyr = df[['Review', 'Rating']]

spotyr.head()

# checking for null values
df.isnull().sum()

# converting scores to sentiment
def to_sentiment(rating):

    rating = int(rating)

    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

spotyr['Rating'] = df.Rating.apply(to_sentiment)

batchspoty = spotyr[:2500]

batchspoty.head()

# turning into a string
batchspoty['Review'] = batchspoty['Review'].astype(str)

# making it lowercase
batchspoty["Review"] = batchspoty["Review"].str.lower()

# checking for duplicates
print("Number of duplicates: " + str(batchspoty.duplicated().sum()))

# remove duplicate
batchspoty.drop_duplicates(inplace = True)

# check for null values
batchspoty.isnull().sum()

# remove punctuation
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

batchspoty["Review"] = batchspoty["Review"].apply(lambda text: remove_punctuation(text))

# stopwords list & removal
STOPWORDS_CUSTOM = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def remove_stopwords_custom(text):
    return " ".join([word for word in str(text).split() if word.lower() not in STOPWORDS_CUSTOM])

batchspoty["Review"] = batchspoty["Review"].apply(remove_stopwords_custom)

batchspoty.head()

# lemmatization & stemming
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def stemming(sentence):
    tokens = nltk.word_tokenize(sentence)
    stems = [stemmer.stem(token) for token in tokens]
    new_sentence = " ".join(stems)
    return new_sentence


def lemma(sentence):
    tokens = nltk.word_tokenize(sentence)
    lemma = [lemmatizer.lemmatize(token) for token in tokens]
    new_sentence = " ".join(lemma)
    return new_sentence



batchspoty["lemma"] = batchspoty['Review'].apply(lemma)
batchspoty["stem"] = batchspoty['Review'].apply(stemming)

batchspoty

# Let's now load a pre-trained BERT model.
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# tokenization
tokenized = batchspoty['Review'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# padding
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

np.array(padded).shape

#attention mask
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

# deep learning
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

cdfeatures = last_hidden_states[0][:,0,:].numpy()

labels = batchspoty['Rating']

train_features, test_features, train_labels, test_labels = train_test_split(cdfeatures, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

lr_clf.score(test_features, test_labels)

# comparing to other models
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_features, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()

scores_rf = cross_val_score(clf_rf, train_features, train_labels)
print("RandomForestClassifier score: %0.3f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()

scores_dt = cross_val_score(clf_dt, train_features, train_labels)
print("DecisionTreeClassifier score: %0.3f (+/- %0.2f)" % (scores_dt.mean(), scores_dt.std() * 2))

# visualisation
sns.histplot(df['Rating'], bins = 15)

text = ' '.join(spotyr['Review'])
wordcloud = WordCloud(width=800, height=400, max_words=150, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.show()

sns.barplot(x='Rating', y='Review_Length', data=df, ci='sd')
plt.title('Review Length vs Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Mean Review Length')
plt.show()

negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nowhere', 'nothing', 'neither', 'nor']

df['Negation_Count'] = df['Review'].apply(lambda x: sum(1 for word in x.split() if word.lower() in negation_words))

plt.figure(figsize=(10, 6))
plt.hist(df['Negation_Count'], bins=range(11), align='left', edgecolor='black')
plt.title('Distribution of Negation Counts in Reviews')
plt.xlabel('Number of Negations')
plt.ylabel('Frequency')
plt.show()

average_thumbs_up = df.groupby('Rating')['Total_thumbsup'].mean()

plt.figure(figsize=(8, 5))
average_thumbs_up.plot(kind='bar', color='green', edgecolor='black')
plt.title('Average Thumbs-Up by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Thumbs-Up')
plt.xticks(rotation=0)
plt.show()

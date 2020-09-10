from sklearn.datasets import fetch_20newsgroups
from os.path import isfile
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def filter_article(article, filts):
    filts = filts.split()

    if 'lem' in filts:
        article = [lemmatizer.lemmatize(x, pos='v') for x in article]

    if 'stop' in filts:
        article = [x for x in article if x not in stopwords.words('english')]

    if 'stem' in filts:
        article = [stemmer.stem(x) for x in article]

    if 'filt' in filts:
        article = [x for x in article if x.isalpha()]

    return article


def filter_articles(name, articles, filts):
    filename = f'{name}-{preproc}.p'
    if os.path.isfile(filename):
        with open(filename, 'rb') as fp:
            filtered_articles = pickle.load(fp)

    else:
        filtered_articles = []
        for data in articles:
            tok = word_tokenize(data)
            filtered_articles.append(filter_article(tok, filts))

        with open(filename, 'wb') as fp:
            pickle.dump(filtered_articles, fp)

    return filtered_articles


TT_FILE = 'twenty-train.p'
if isfile(TT_FILE):
    twenty_train = pickle.load(open(TT_FILE, 'rb'))
else:
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    pickle.dump(twenty_train, open(TT_FILE, 'wb'))

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stemmer = PorterStemmer()

ans_fmt = """Preprocesamiento: {preproc}
Longitud del vocabulario: {vocab_len}
"""
preprocessing = ['tok', 'tok lem stem', 'tok stop', 'tok lem stop stem', 'tok lem stop stem filt']

for preproc in preprocessing:
    filtered_articles = filter_articles('train', twenty_train.data, preproc)
    vocab = set([word for article in filtered_articles for word in article])
    print(ans_fmt.format(preproc=preproc, vocab_len=len(vocab)))

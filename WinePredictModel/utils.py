import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
nltk.download('stopwords')
nltk.download('movie_reviews')

def clean_descriptions(df, description_column):
    all_stops = set(stopwords.words('english')).union(set(string.punctuation))
    clean_desc =[]
    for sentence in list(df[description_column]):
        tok_desc = word_tokenize(sentence)
        lower_data = [tok.lower() for tok in tok_desc]
        tok_desc_no_num = [i for i in lower_data if i.isalpha()]
        clean_desc.append([i for i in tok_desc_no_num if i not in all_stops])
    return [' '.join(i) for i clean_desc]

def clean_description_sentiment(df,description_column):
   cleaned_description = clean_descriptions(df,description_column)
   tb = Blobber(analyzer=NaiveBayesAnalyzer())
   sentiment_values = [text.sentiment for text in blob]
   sentiment_df = pd.DataFrame(zip(*sentiment_values)).T.drop(columns='clf')
   return df.join(sentiment_df)

def vocab_richness(text):
    tokens = word_tokenize(text)
    total_length = len(tokens)
    unique_words = set(tokens)
    unique_word_length = len(unique_words)
    return unique_word_length/total_length

def f1(model,X_test,y_test):
    y_pred = model.predict(X_test)
    score = f1_score(y_test,y_pred,average='weighted')
    return score

def create_dummies_ohe(X,cat_columns):
    categorical_x = pd.get_dummies(X[cat_columns],prefix=['1','2','3','4','5'])
    numeric_cols = list(set(X.columns)-set(cat_columns))
    return pd.concat([X[numeric_cols],categorical_x], axis=1)





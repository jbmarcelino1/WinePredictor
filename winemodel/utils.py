import string
# from nltk.tokenize import word_tokenize
# import nltk
# from nltk.corpus import stopwords
# from textblob import Blobber
# from textblob.sentiments import NaiveBayesAnalyzer
from WinePredictModel.data import GetData
import pandas as pd
from sklearn.metrics import f1_score

# nltk.download("stopwords")
# nltk.download("movie_reviews")
# nltk.download("punkt")


# def clean_descriptions(df, description_column):
#     all_stops = set(stopwords.words("english")).union(set(string.punctuation))
#     clean_desc = []
#     for sentence in list(df[description_column]):
#         tok_desc = word_tokenize(sentence)
#         lower_data = [tok.lower() for tok in tok_desc]
#         tok_desc_no_num = [i for i in lower_data if i.isalpha()]
#         clean_desc.append([i for i in tok_desc_no_num if i not in all_stops])
#     return [" ".join(i) for i in clean_desc]


# def clean_description_sentiment(df, description_column):
#     cleaned_description = clean_descriptions(df, description_column)
#     tb = Blobber(analyzer=NaiveBayesAnalyzer())
#     blob = [tb(text) for text in cleaned_description]
#     sentiment_values = [text.sentiment for text in blob]
#     sentiment_df = pd.DataFrame(zip(*sentiment_values)).T
#     sentiment_df.columns = ["clf", "pos", "neg"]
#     sentiment_df = sentiment_df[["pos", "neg"]].astype(float)
#     return df.join(sentiment_df)


# def vocab_richness(text):
#     tokens = word_tokenize(text)
#     total_length = len(tokens)
#     unique_words = set(tokens)
#     unique_word_length = len(unique_words)
#     return unique_word_length / total_length


def f1(model, X_test, y_test):
    print(X_test)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average="weighted")
    return score


def create_dummies_ohe(X, cat_columns):
    categorical_x = pd.get_dummies(
        X[cat_columns], prefix=[str(i) for i in range(len(cat_columns))]
    )
    numeric_cols = list(set(X.columns) - set(cat_columns))
    return pd.concat([X[numeric_cols], categorical_x], axis=1)


def select_cat_data_threshold(X, feature_data, threshold_value, cat_features):
    dictionary_filter = {
        cat: feature_data.loc[
            (feature_data[cat] == True) & (feature_data["scores"] <= threshold_value),
            "features",
        ]
        for cat in cat_features
    }
    for cat, series in dictionary_filter.items():
        X.loc[X[cat].isin(series), cat] = "Other"
    return X

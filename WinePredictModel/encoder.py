import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from WinePredictModel.utils import (
    clean_descriptions,
    clean_description_sentiment,
    vocab_richness,
    select_cat_data_threshold,
    create_dummies_ohe
)
from sklearn.impute import SimpleImputer
from WinePredictModel.data import GetData
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
TEMP = "temperature"
COUNTRY_ISO = "country_iso_data"
WEATHER_MONTH = "weather_country_month_v2"
FEATURES = "data_test"
FILE_LOCATION = "gcp"
CAT_FEATURES = ["province", "variety", "country", "winery", "region_1"]


class YearVintageEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, title):
        self.title = title

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X['year'] = X[self.title].str.extract('(\d+)')
        X["year"] = pd.to_numeric(X["year"])
        X["year"] = np.where(
            (X["year"] >= 2021) | (X["year"] <= (2021 - 70)), np.nan, X["year"]
        )
        X["year"] = pd.to_datetime(X["year"],format='%Y').dt.year
        sc = SimpleImputer(strategy = 'median')
        X[['year']] = sc.fit_transform(X[['year']])
        return X

    def fit(self, X, y=None):
        return self

class YearReturnEnconder(BaseEstimator, TransformerMixin):
    def __init__(self, year):
        self.year = year

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        return X[[self.year]]

    def fit(self, X, y=None):
        return self


class DescriptionSentimentEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, description):
        self.description = description
        self.simp_imp = SimpleImputer(strategy='median')

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X = clean_description_sentiment(X, self.description)
        X[["pos", "neg"]] = self.simp_imp.fit_transform(X[["pos", "neg"]])
        X[["pos", "neg"]] = X[["pos", "neg"]].astype(float)
        return X[["pos", "neg"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class VocabRichnessEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, description):
        self.description = description

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X["vocab richness"] = X[self.description].apply(vocab_richness)
        return X[["vocab richness"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class TitleLengthEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, taster_name, title):
        self.taster_name = taster_name
        self.title = title

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        col_list = [self.title, self.taster_name]
        X[col_list] = X[col_list].astype(str)
        for col in col_list:
            X[f"{col}_length"] = X[col].apply(lambda x: len(x))
        return X[["title_length", "taster_name_length"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class PriceBinEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, price):
        self.price = price

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X["price_bin"] = pd.cut(X[self.price], bins=15, labels=False)
        return X[["price_bin"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class WeatherEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, country, year):
        self.country = country
        self.year = year
        d = GetData(FILE_LOCATION)
        self.temp = d.select_data_type(TEMP)
        self.country_iso = d.select_data_type(COUNTRY_ISO)
        self.weather_month = d.select_data_type(WEATHER_MONTH)

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        weather_feature = X.set_index(self.country).join(
            self.country_iso.set_index("country")
        )
        weather_iso_df = self.weather_month.set_index("country").join(
            self.country_iso.set_index("country")
        )
        weather_iso_df["year"] = pd.to_datetime(weather_iso_df["month"]).dt.year
        weather_iso_summary_df = weather_iso_df.groupby(
            ["country_iso", "year"], as_index=False
        ).mean()
        df = pd.merge(
            weather_feature,
            weather_iso_summary_df,
            how="left",
            left_on=["country_iso", "year"],
            right_on=["country_iso", "year"],
        )
        return df[['avg_temp']].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class PriceImputer(BaseEstimator, TransformerMixin):
    def __init__(self,price):
        self.price = price

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        si = SimpleImputer(strategy='median')
        X[[self.price]] = si.fit_transform(X[[self.price]])

        return X

    def fit(self, X, y=None):
        return self

class CreateDummies(BaseEstimator, TransformerMixin):
    def __init__(self,cat_features=CAT_FEATURES):
        self.cat_features = cat_features

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        prefix_list = [str(i) for i in range(len(self.cat_features))]
        categorical_x = pd.get_dummies(X[self.cat_features], prefix=prefix_list)
        return categorical_x

    def fit(self, X, y=None):
        return self


class FeatureSelectionEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, cat_features=CAT_FEATURES):
        d = GetData(FILE_LOCATION)
        self.features = d.select_data_type(FEATURES)
        self.threshold = threshold
        self.cat_features = cat_features

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        self.features.columns = [i.replace('country_iso','country') for i in self.features.columns]
        X_filtered =  select_cat_data_threshold(
            X, self.features, self.threshold, self.cat_features
        )
        return X_filtered.reset_index(drop=True)

    def fit(self, X, y=None):
        return self


if __name__ == "__main__":
    df = GetData('gcp').clean_data()
    fs = FeatureSelectionEncoder(1E-6)
    df = fs.fit_transform(df)
    print(df.columns)
    # dist = YearVintageEncoder('title')
    # X_dist = dist.transform(df)
    # assert isinstance(X_dist,pd.DataFrame)
    # assert "year" in X_dist.columns, 'should contain year'
    # sent = DescriptionSentimentEncoder('description')
    # X_sent = sent.transform(df)
    # assert isinstance(X_sent,pd.DataFrame)
    # print(X_sent["pos"].dtype)
    # vocab = VocabRichnessEncoder('description')
    # X_vocab = vocab.transform(df)
    # assert  isinstance(X_vocab,pd.DataFrame)
    # print(X_vocab["vocab richness"].dtype)
    # title_len = TitleLengthEncoder('taster_name','title')
    # X_len = title_len.transform(df)
    # print(X_len.head())
    # pb = PriceBinEncoder('price')
    # X_pb = pb.transform(df)
    # print(len(X_pb["price_bin"].unique()))
    # we = WeatherEncoder('country','year')
    # X_we = we.transform(df)
    # print(X_we.columns)
    # print(X_we.info())
    # fs = FeatureSelectionEncoder(1E-6)
    # X_features = fs.transform(df)
    # print(X_features.columns)
    # print(X_features.info())

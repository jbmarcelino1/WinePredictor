import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from  WinePredictModel.utils import (clean_descriptions,
    clean_description_sentiment,
    vocab_richness)
from WinePredictModel.data import GetData
TEMP = 'temperature'
COUNTRY_ISO = 'country_iso_data'
WEATHER_MONTH = 'weather_country_month_v2'


class YearVintageEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, title):
        self.title = title

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X['year'] = pd.to_numeric(X[self.title].str.extract('(\d+)'))
        X['year'] = np.where(
                    (X['year']>=2021) | (X['year']<=(2021-70)),
                    np.nan,
                    X['year'])
        X['year'] = pd.to_datetime(X['year']).dt.year
        return X[["year"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self

class DescriptionSentimentEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, description):
        self.description = description

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X = clean_description_sentiment(X,self.description)
        return X[['pos','neg']].reset_index(drop=True)

    def fit(self, X, y=None):
        return self

class VocabRichnessEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, description):
        self.description = description

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X['vocab richness'] = X[self.description].apply(vocab_richness)
        return X[['vocab richness']].reset_index(drop=True)

    def fit(self, X, y=None):
        return self

class TitleLengthEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, taster_name, title):
        self.taster_name = taster_name
        self.title = title

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        for col in [self.title,self.taster_name]:
            X[f"{col}_length"] = X[col].apply(lambda x: len(x))
        return X[['title_length','taster_name_length']].reset_index(drop=True)

    def fit(self, X, y=None):
        return self

class PriceBinEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, price):
        self.price = price

    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        X['price_bin'] = pd.cut(X['price'],bins=15,labels=False)
        return X[['price_bin']].reset_index(drop=True)

    def fit(self, X, y=None):
        return self

class WeatherEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, country):
        self.country = country
        d = GetData("mac")
        self.temp = d.select_data_type(TEMP)
        self.country_iso = d.select_data_type(COUNTRY_ISO)
        self.weather_month = d.select_data_type(WEATHER_MONTH)


    def transform(self, X, y=None):
        """implement encode here"""
        assert isinstance(X, pd.DataFrame)
        weather_feature = X.set_index(self.country).join(self.country_iso.set_index('country'))
        weather_iso_df = self.weather_month.set_index('country').join(self.country_iso.set_index('country'))
        weather_iso_df['year'] = pd.to_datetime(weather_iso_df['month']).dt.year
        weather_iso_summary_df = weather_iso_df.groupby(['country_iso', 'year'], as_index=False).mean()
        df = pd.merge(
              weather_feature,
              weather_iso_summary_df,
              how='left',
              left_on=['country_iso','year'],
              right_on=['country_iso','year']
             )
        return df

    def fit(self, X, y=None):
        return self

# TODO: add feature remover function

if __name__ == "__main__":
    params =
    df = get_data(**params)
    df = clean_df(df)
    dist = DistanceTransformer()
    X = dist.transform(df)


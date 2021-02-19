import multiprocessing
import time
import warnings
import joblib
import mlflow
import pandas as pd
from tempfile import mkdtemp
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from termcolor import colored
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
from data import GetData
from encoder import (
    YearVintageEncoder,
    DescriptionSentimentEncoder,
    VocabRichnessEncoder,
    TitleLengthEncoder,
    PriceBinEncoder,
    WeatherEncoder,
    FeatureSelectionEncoder,
    YearReturnEnconder,
    PriceImputer,
    CreateDummies,
)
from utils import f1
from gcp import storage_upload
import numpy as np
from sklearn.metrics import f1_score

THRESHOLD = 1e-6
MLFLOW_URI = "https://mlflow.lewagon.co/"
CAT_FEATURES = ["province", "variety", "country", "winery", "region_1"]
MODEL_VERSION = 'Pipeline'

class Trainer(object):
    ESTIMATOR = "RandomForest"
    EXPERIMENT_NAME = "WinePredictionModlel"

    def __init__(self, X, y, **kwargs):
        # TODO: add a doctsring
        self.pipeline_feature = None
        self.kwargs = kwargs
        self.local = kwargs.get("local", False)
        self.mlflow = kwargs.get("mlflow", False)
        self.pca = kwargs.get('pca',False)
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)
        self.model_params = None  # for
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.3, random_state=2
            )
        self.log_kwargs_params()
        self.log_machine_specs()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "RandomForest":
            model = RandomForestClassifier()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'bootstrap': [True, False],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            }
        if estimator == "xgboost":
            model = XGBClassifier(
                objective="multi:softmax",
                n_jobs=self.n_jobs,
                max_depth=10,
                learning_rate=0.05,
                gamma=3,
            )
            self.model_params = {
                "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 17],
                "min_child_weight": [1, 3, 5, 7, 9, 11],
                "gamma": [0.0, 0.1, 0.3, 0.4, 0.5, 0.6],
                "colsample_bytree": [0.3, 0.4, 0.5, 0.7, 0.8, 0.9],
                "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            }
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        return model

    def get_pca_type(self):
        pca_type = self.kwargs.get('pca_type',None)
        if pca_type  == 'PCA':
            pca = PCA()
            self.pca_params = {
             "n_components":range(9, 20)
            }
        if pca_type == 'KernelPCA':
            pca = KernelPCA()
            self.pca_params = {
             "n_components":range(9, 20),
             "kernel": ('linear', 'poly', 'rbf','cosine')
            }
        pca_params = self.kwargs.get("pca_params", {})
        self.mlflow_log_param("pca", pca)
        pca.set_params(**pca_params)
        return pca

    def set_pipeline(self):
        memory = self.kwargs.get("pipeline_memory", None)
        feateng_steps = self.kwargs.get(
            "feateng", [
                        "year",
                        "weather",
                        "description_sentiment",
                        "title_length",
                        "categorical",
                        "price_quan",
                        "vocab_richness",
                        "price_bin"
                        ]
        )
        if memory:
            memory = mkdtemp()
        # define feature selection columntransformer
        pipe_sentiment = make_pipeline(
            DescriptionSentimentEncoder(description="description"),
            QuantileTransformer(),
        )
        pipe_title_length = make_pipeline(
            TitleLengthEncoder(taster_name="taster_name", title="title"),
            QuantileTransformer(),
        )
        pipe_vocab_richness = make_pipeline(
            VocabRichnessEncoder(description="description"), QuantileTransformer()
        )
        price_bin = make_pipeline(PriceBinEncoder(price="price"), OneHotEncoder(handle_unknown='ignore'))

        pipe_weather = make_pipeline(
                    WeatherEncoder('country','year'),
                    SimpleImputer(strategy = 'median'),
                    QuantileTransformer()
                )
        # Define default feature engineering blocs
        feateng_blocks = [
            ("weather", pipe_weather, ["country","year"]),
            ("year",YearReturnEnconder("year"),["year"]),
            ("price_quan",QuantileTransformer(),["price"]),
            ("description_sentiment", pipe_sentiment, ["description"]),
            ("title_length", pipe_title_length, ["taster_name","title"]),
            ("vocab_richness", pipe_vocab_richness, ["description"]),
            ("price_bin", price_bin, ["price"]),
            ("categorical",OneHotEncoder(handle_unknown='ignore'),CAT_FEATURES)]

        features_encoder = ColumnTransformer(
                    feateng_blocks, n_jobs=None,
                    remainder='drop'
                )

        #Filter out some bocks according to input parameters
        for bloc in feateng_blocks:
            if bloc[0] not in feateng_steps:
                feateng_blocks.remove(bloc)


        self.pipeline_feature = Pipeline(
            steps=[
                ('year', YearVintageEncoder(title="title")),
                ('price_impute',PriceImputer(price='price')),
                ('feat_eng',features_encoder),
            ],
            memory=memory,
        )

        # if self.pca:
        #      self.pipeline.steps.insert(-2, ["pca", PCA(n_components=12)])

    def train(self, gridsearch=False):
        tic = time.time()
        self.set_pipeline()
        X_train_preproc = self.pipeline_feature.fit_transform(self.X_train)
        bm = BorderlineSMOTE(random_state=2,sampling_strategy='minority',k_neighbors=1,m_neighbors=20)
        self.X_train_smote,self.y_train_smote = bm.fit_resample(X_train_preproc,self.y_train)
        self.model = self.get_estimator()
        if gridsearch:
            self.model = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.model,
            n_iter=10,
            cv=2,
            verbose=1,
            random_state=42,
            n_jobs=None,
            )
            self.model.fit(self.X_train_smote,self.y_train_smote)
            self.mlflow_log_metric("train_time", int(time.time() - tic))
        else:
            self.model.fit(self.X_train_smote,self.y_train_smote)
            self.mlflow_log_metric("train_time", int(time.time() - tic))

    def compute_f1(self, X_test, y_test):
        if self.pipeline_feature is None:
            raise ("Cannot evaluate an empty pipeline")
        X_test_preproc = self.pipeline_feature.transform(X_test)
        y_pred = self.model.predict(X_test_preproc)
        f1_sc = f1_score(y_test,y_pred,average="weighted")
        return round(f1_sc, 3)

    def evaluate(self):
        f1_val = self.compute_f1(self.X_val, self.y_val)
        self.mlflow_log_metric("f1_val", f1_val)
        print(
            colored("f1 val: {}".format(f1_val), "blue")
        )

    def save_model(self, upload=True, auto_remove=True):
        if self.local:
            joblib.dump(self.pipeline_feature, "feature_eng.joblib")
            joblib.dump(self.model,"model.joblib")
        if not self.local:
            storage_upload(model_version=MODEL_VERSION)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        clf = self.get_estimator()
        self.mlflow_log_param("estimator_name", clf.__class__.__name__)
        params = clf.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    feat_eng = [
                "year",
                "weather",
                "description_sentiment",
                "categorical",
                "price_quan",
                "vocab_richness",
                "price_bin"
                        ]
    experiment = "winereviewpredict_set_EB"
    params = dict(
                  upload=True,
                  gridsearch=False,
                  estimator="RandomForest",
                  mlflow=True,
                  local=True,  # set to True to log params to mlflow
                  experiment_name=experiment,
                  estimator_params={'n_estimators':500},
                  feateng=feat_eng)
    print("############   Loading Data   ############")
    d = GetData("gcp",nrows=5000)
    df = d.clean_data()
    y_train = df["points"]
    X_train = df.drop("points", axis=1)
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    # print(colored("############   Saving model    ############", "green"))
    # t.save_model()

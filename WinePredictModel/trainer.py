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
    CreateDummies
)
from utils import f1

THRESHOLD = 1e-6
MLFLOW_URI = "https://mlflow.lewagon.co/"
CAT_FEATURES = ["province", "variety", "country", "winery", "region_1"]
MODEL_VERSION = 'Pipeline'

class Trainer(object):
    ESTIMATOR = "RandomForest"
    EXPERIMENT_NAME = "WinePredictionModlel"

    def __init__(self, X, y, **kwargs):
        # TODO: add a doctsring
        self.pipeline = None
        self.kwargs = kwargs
        self.local = kwargs.get("local", False)
        self.mlflow = kwargs.get("mlflow", False)
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)
        self.model_params = None  # for
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.15, random_state=1
            )
        # self.pca = self.kwargs.get
        self.log_kwargs_params()
        self.log_machine_specs()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "RandomForest":
            model = RandomForestClassifier()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                "max_features": ["auto", "sqrt"],
                "n_estimators": range(60, 220, 40),
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
                "n_estimators": range(60, 220, 40),
            }
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        return model

    def set_pipeline(self):
        memory = self.kwargs.get("pipeline_memory", None)
        feateng_steps = self.kwargs.get(
            "feateng", ["year", "weather", "description_sentiment", "title_length","categorical"]
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
        price_bin = make_pipeline(PriceBinEncoder(price="price"), OneHotEncoder())

        pipe_weather = make_pipeline(
                    WeatherEncoder('country','year'),
                    SimpleImputer(strategy = 'median'),
                    QuantileTransformer()
                )
        # Define default feature engineering blocs
        feateng_blocks = [
            ("weather", pipe_weather, ["country","year"]),
            ("year",YearReturnEnconder("year"),["year"]),
            ("description_sentiment", pipe_sentiment, ["description"]),
            ("title_length", pipe_title_length, ["taster_name","title"]),
            ("vocab_richness", pipe_vocab_richness, ["description"]),
            ("price_bin", price_bin, ["price"]),
            ("categorical",CreateDummies(),CAT_FEATURES)
        ]
        # Filter out some bocks according to input parameters
        for bloc in feateng_blocks:
            if bloc[0] not in feateng_steps:
                feateng_blocks.remove(bloc)

        features_encoder = ColumnTransformer(
            feateng_blocks, n_jobs=None,
            remainder='drop'
        )

        self.pipeline = Pipeline(
            steps=[
                ('feature',FeatureSelectionEncoder(threshold=1E-6)),
                ('year', YearVintageEncoder(title="title")),
                ('price_impute',PriceImputer(price='price')),
                ('feat_eng',features_encoder),
                ('scaler',QuantileTransformer()),
                ("clf", self.get_estimator())
            ],
            memory=memory,
        )

        # if self.pca:
        #     self.pipeline.steps.insert(-2, ["pca", PCA(n_components=12)])

        # TODO add optimze function similar to taxifaremodel

    def add_grid_search(self):
        # Here to apply ramdom search to pipeline, need to follow naming "rgs__paramname"
        params = {"clf__" + k: v for k, v in self.model_params.items()}
        self.pipeline = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=params,
            n_iter=10,
            cv=2,
            verbose=1,
            random_state=42,
            n_jobs=None,
        )

    def train(self, gridsearch=False):
        tic = time.time()
        self.set_pipeline()
        if gridsearch:
            self.add_grid_search()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def compute_f1(self, X_test, y_test):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        f1_score = f1(self.pipeline, X_test, y_test)
        return round(f1_score, 3)

    def evaluate(self):
        f1_train = self.compute_f1(self.X_train, self.y_train)
        self.mlflow_log_metric("f1_train", f1_train)
        if self.split:
            f1_val = self.compute_f1(self.X_val, self.y_val)
            self.mlflow_log_metric("f1_val", f1_val)
            print(
                colored("f1 train: {} || f1 val: {}".format(f1_train, f1_val), "blue")
            )
        else:
            print(colored("f1 train: {}".format(f1_train), "blue"))

    def save_model(self, upload=True, auto_remove=True):
        if self.local:
            joblib.dump(self.pipeline, "model.joblib")
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
    experiment = "winereviewpredict_set_EB"
    params = dict(
                  upload=True,
                  gridsearch=False,
                  estimator="RandomForest",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment)
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
    # del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()

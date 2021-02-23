import joblib
import os
FILEPATH = r'model'
FEATURE = 'feature_eng'
MODEL = 'model'

def load_models(base_filepath=FILEPATH):
    joblib_dict = dict()
    for model in os.listdir(base_filepath):
        if model.endswith(".joblib"):
            joblib_dict[model.split(".")[0]] = joblib.load(
                os.path.join(base_filepath,model))
    return joblib_dict.get(FEATURE),joblib_dict.get(MODEL)


if __name__ == '__main__':
    test1, test2 = load_models()




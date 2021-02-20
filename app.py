import flask
from flask import request, escape
import pandas as pd
import joblib
MEDIAN_PRICE = 27
app = flask.Flask(__name__)

@app.route('/winemodel',methods=['GET','POST'])
def predict_wine_rating():
    country = request.args.get('country')

    description = request.args.get('description','wine')
    price = float(request.args.get('price',str(MEDIAN_PRICE)))
    province = request.args.get('province')
    region = 'Other'
    variety = request.args.get('variety')
    taster_name = request.args.get('taster_name')
    year = request.args.get('year')
    winery = request.args.get('winery')
    title = f"{winery} {year} {variety} ({region})"

    df = pd.DataFrame(dict(country=[country],
                description=[description],
                price=[price],
                province=[province],
                region_1=[region],
                taster_name=[taster_name],
                title=[title],
                variety=[variety],
                winery=[winery]
                ))
    feat = joblib.load("model/feature_eng.joblib")
    model = joblib.load("model/model.joblib")
    feat_eng_x = feat.transform(df)
    prediction = model.predict(feat_eng_x)
    return {'prediction':str(prediction[0])}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

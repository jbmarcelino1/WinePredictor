import flask
from flask import request, escape
import pandas as pd
import joblib
from currency_converter import CurrencyConverter
app = flask.Flask(__name__)

@app.route('/winemodel',methods=['GET','POST'])
def predict_wine_rating():
    country = request.args.get('country').lower().title()
    description = request.args.get('description','wine')
    # conversion rates - create a drop down with EUR and GBR
    conversion_factor = request.args.get('conversion','EUR')
    c = CurrencyConverter()
    price_raw = float(request.args.get('price'))
    price = c.convert(price_raw, conversion_factor, 'USD')
    # province always has to be provided
    province = request.args.get('province')
    # region defaults to Other in all cases
    region = 'Other'
    variety = request.args.get('variety')
    year = request.args.get('year')
    winery = request.args.get('winery')
    title = f"{winery} {year} {variety} ({province})"
    # dataframe has to be in this format
    df = pd.DataFrame(dict(country=[country],
                description=[description],
                price=[price],
                province=[province],
                region_1=[region],
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

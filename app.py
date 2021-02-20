import flask
from flask import request, escape
MEDIAN_PRICE = 27
app = flask.Flask(__name__)

@app.route('/winemodel',method=['GET'])
def predict_wine_rating():
    country = request.args.get('country')
    description = request.args.get('description','wine')
    price = float(request.args.get('price',str(MEDIAN_PRICE)))
    province = request.args.get('province','Other')
    region = request.args.get('region')
    variety = request.args.get('variety')
    #TODO remove later
    taster_name = request.args.get('taster_name')
    year = request.args.get('year')
    # if brand and winery the same
    if 'brand' in request.args:
    brand = request.args.get('brand')
    winery = request.args.get('winery')
    # if winery different

    if 'brand' in request.args and 'winery'in request.args:

        title = f"{brand} {year} {variety} ({region})"



    test = int(request.args.get('age',0))
    try:
        return f"you are {test}"
    except KeyError:
        return f'{escape(test)} invalid input'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

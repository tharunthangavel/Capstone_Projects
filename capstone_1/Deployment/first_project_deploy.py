from flask  import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

autoprice_model = pickle.load(open('auto_rf_regression_model.pkl', 'rb'))

column_transformer = pickle.load(open('categorical_encoding_transformer.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('autoprice.html')

@app.route("/predict", methods=['POST'])
def predict():
    fuel_type = request.form['fuel_type']
    aspiration = request.form['aspiration']
    body_style = request.form['body-style']
    drive_wheels = request.form['drive-wheels']
    engine_location = request.form['engine-location']
    num_of_cylinders = request.form['num-of-cylinders']
    fuel_system = request.form['fuel-system']
    engine_type = request.form['engine-type']
    make = request.form['make']
    num_of_doors = request.form['num-of-doors']
    wheel_base = float(request.form['wheel-base'])
    curb_weight = float(request.form['curb-weight'])
    engine_size = float(request.form['engine-size'])
    compression_ratio = float(request.form['compression-ratio'])
    horsepower = float(request.form['horsepower'])
    peak_rpm = float(request.form['peak-rpm'])
    highway_mpg = float(request.form['highway-mpg'])
    bore = float(request.form['bore'])
    stroke = float(request.form['stroke'])   
    length = float(request.form['length'])
    width = float(request.form['width'])
    height = float(request.form['height'])

    if fuel_system == 'mpfi':
        fuel_system = 'mfi'
    
    if fuel_system == 'spdi':
        fuel_system = 'spfi'
    
    bore_stroke_ratio = bore / stroke

    volume = (length * width * height)

    data_dict = {'make':[make], 'fuel-type':[fuel_type], 'aspiration':[aspiration], 'num-of-doors':[num_of_doors], 
       'body-style':[body_style],'drive-wheels':[drive_wheels], 'engine-location':[engine_location], 'wheel-base':[wheel_base], 
       'curb-weight':[curb_weight],'engine-type':[engine_type] , 'num-of-cylinders':[num_of_cylinders], 'engine-size':[engine_size], 
       'fuel-system':[fuel_system],'compression-ratio':[compression_ratio], 'horsepower':[horsepower], 'peak-rpm':[peak_rpm], 
       'highway-mpg':[highway_mpg],'bore_stroke_ratio':[bore_stroke_ratio], 'volume':[volume]}

    data_df = pd.DataFrame(data_dict)

    X_user_transformed = column_transformer.transform(data_df)

    price_prediction = autoprice_model.predict(X_user_transformed)
    predicted_value = round(price_prediction[0],2)
    
    return render_template('autoprice.html',pred=predicted_value)

if __name__ == '__main__':
    app.run()



from flask  import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
#import requests
import pickle
app = Flask(__name__)
autoprice_model = pickle.load(open('auto_rf_regression_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('autoprice.html')

@app.route("/predict", methods=['POST'])
def predict():
    pd.set_option('display.max_columns', 500)
    
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
    bore_stroke_ratio = bore*stroke
        
    length = float(request.form['length'])
    width = float(request.form['width'])
    height = float(request.form['height'])
    volume = length*width*height

    data_dict = {'make':[make], 'fuel-type':[fuel_type], 'aspiration':[aspiration], 'num-of-doors':[num_of_doors], 'body-style':[body_style],
       'drive-wheels':[drive_wheels], 'engine-location':[engine_location], 'wheel-base':[wheel_base], 'curb-weight':[curb_weight],
       'engine-type':[engine_type] , 'num-of-cylinders':[num_of_cylinders], 'engine-size':[engine_size], 'fuel-system':[fuel_system],
       'compression-ratio':[compression_ratio], 'horsepower':[horsepower], 'peak-rpm':[peak_rpm], 'highway-mpg':[highway_mpg],
       'bore_stroke_ratio':[bore_stroke_ratio], 'volume':[volume]}
    data_df = pd.DataFrame(data_dict)
    print('data_df =',data_df)
    print('data_df.size = ',data_df.size)


    X_test_transform = transform_values(data_df,num_of_doors,engine_type)
    print('X_test_transform =',X_test_transform)
    print('X_test_transform.size = ',X_test_transform.size)

    price_prediction = autoprice_model.predict(X_test_transform)
    pred = price_prediction[0]
    pred=round(pred,2)
    return render_template('autoprice.html',pred=pred)




def transform_values(data_df,num_of_doors,engine_type):
    print('Engine Type:', engine_type)
    print('Type of Engine Type:', type(engine_type))
    
    transformer = ColumnTransformer(transformers=[
    ('tnf1',OrdinalEncoder(categories=[['gas','diesel']]),['fuel-type']),
    ('tnf2',OrdinalEncoder(categories=[['std','turbo']]),['aspiration']),
    ('tnf3',OrdinalEncoder(categories=[['hatchback','wagon','sedan','hardtop','convertible']]),['body-style']),
    ('tnf4',OrdinalEncoder(categories=[['fwd','4wd','rwd']]),['drive-wheels']),                                                      
    ('tnf5',OrdinalEncoder(categories=[['front','rear']]),['engine-location']),
    ('tnf6',OrdinalEncoder(categories=[['four','six','five','two','eight','three','twelve']]),['num-of-cylinders']),
    ('tnf7',OrdinalEncoder(categories=[['mpfi','2bbl','idi','1bbl','spdi','4bbl']]),['fuel-system']),
    ('tnf8',OrdinalEncoder(categories=[['mercedes-benz','bmw','porsche','jaguar','audi','volvo','nissan','saab','mazda','peugot','toyota','mercury','alfa-romero','mitsubishi','volkswagen','dodge','honda','plymouth','subaru','isuzu','renault','chevrolet']]),['make']),
    ('tnf9',OneHotEncoder(sparse=True,drop='first',handle_unknown='error'),['num-of-doors','engine-type'])],remainder='passthrough')
   
    # For some weird reason the one hot encoder from column transforer is not working and hence below manual encoding
    
    X_test_transform = pd.DataFrame(transformer.fit_transform(data_df), columns=transformer.get_feature_names_out())
    if engine_type=='ohc':
        X_test_transform['tnf9__engine-type_l']=[0]
        X_test_transform['tnf9__engine-type_ohc']=[1]
        X_test_transform['tnf9__engine-type_ohcf']=[0]
        X_test_transform['tnf9__engine-type_ohcv']=[0]
        X_test_transform['tnf9__engine-type_rotor']=[0]
    elif engine_type=='ohcf':
        X_test_transform['tnf9__engine-type_l']=[0]
        X_test_transform['tnf9__engine-type_ohc']=[0]
        X_test_transform['tnf9__engine-type_ohcf']=[1]
        X_test_transform['tnf9__engine-type_ohcv']=[0]
        X_test_transform['tnf9__engine-type_rotor']=[0]
    elif engine_type=='ohcv':
        X_test_transform['tnf9__engine-type_l']=[0]
        X_test_transform['tnf9__engine-type_ohc']=[0]
        X_test_transform['tnf9__engine-type_ohcf']=[0]
        X_test_transform['tnf9__engine-type_ohcv']=[1]
        X_test_transform['tnf9__engine-type_rotor']=[0]
    elif engine_type=='l':
        X_test_transform['tnf9__engine-type_l']=[1]
        X_test_transform['tnf9__engine-type_ohc']=[0]
        X_test_transform['tnf9__engine-type_ohcf']=[0]
        X_test_transform['tnf9__engine-type_ohcv']=[0]
        X_test_transform['tnf9__engine-type_rotor']=[0]
    elif engine_type=='rotor':
        X_test_transform['tnf9__engine-type_l']=[0]
        X_test_transform['tnf9__engine-type_ohc']=[0]
        X_test_transform['tnf9__engine-type_ohcf']=[0]
        X_test_transform['tnf9__engine-type_ohcv']=[0]
        X_test_transform['tnf9__engine-type_rotor']=[1]
    elif engine_type=='dohc':
        X_test_transform['tnf9__engine-type_l']=[0]
        X_test_transform['tnf9__engine-type_ohc']=[0]
        X_test_transform['tnf9__engine-type_ohcf']=[0]
        X_test_transform['tnf9__engine-type_ohcv']=[0]
        X_test_transform['tnf9__engine-type_rotor']=[0]

    if num_of_doors=='two':
        X_test_transform['tnf9__num-of-doors_two']=[1]
    elif num_of_doors=='four':
        X_test_transform['tnf9__num-of-doors_two']=[0]


    return X_test_transform

    
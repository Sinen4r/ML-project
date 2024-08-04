from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,predictPipeline

from src.logger import logging

application=Flask(__name__)

app=application

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST','GET'])
def predictdataPoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
             gender=request.form.get('gender'),
             race_ethnicity=request.form.get('ethnicity'),
             parental_level_of_education=request.form.get("parental_level_of_education"),
             lunch=request.form.get('lunch'),
             test_preparation_course=request.form.get('test_preparation_course'),
             reading_score=float(request.form.get('reading_score')),
             writing_score=float(request.form.get('writing_score'))
        )
        pred_df=data.getData_as_DataFrame()
        print(pred_df)
        predictPipe=predictPipeline()
        results=predictPipe.predict(pred_df)
        return render_template('home.html',results=results)
    
if __name__=='__main__':
    app.run(host="0.0.0.0",port =5000,debug=True)
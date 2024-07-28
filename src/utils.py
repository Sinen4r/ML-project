import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score

from src.exception import CustomerException
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open  (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomerException(e,sys)


def evaluate_model(xTrain,yTrain,xTest,yTest,models):
    
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(xTrain,yTrain)
            yTrain_pred=model.predict(xTrain)   
            yTest_pred=model.predict(xTest)


            testModelScore=r2_score(yTest,yTest_pred)

            report[list(models.keys())[i]]=testModelScore
        return report
    except Exception as e:
        raise CustomerException(e,sys)
import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomerException
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open  (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomerException(e,sys)


def evaluate_model(xTrain,yTrain,xTest,yTest,models,param):
    
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            g=GridSearchCV(model,para,cv=3)
            g.fit(xTrain,yTrain)
            model.set_params(**g.best_params_)
            model.fit(xTrain,yTrain)
            
            yTrain_pred=model.predict(xTrain)   
            yTest_pred=model.predict(xTest)


            testModelScore=r2_score(yTest,yTest_pred)

            report[list(models.keys())[i]]=testModelScore
        return report
    except Exception as e:
        raise CustomerException(e,sys)
    

def load_objects(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomerException(e,sys)
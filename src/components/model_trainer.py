import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomerException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","modell.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_Trainer_config=ModelTrainerConfig()

    def intitiateModelTrainer(self,trainArray,TestArray):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                trainArray[:,:-1],
                trainArray[:,-1],
                TestArray[:,:-1],
                TestArray[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor()
            }
            model_report:dict=evaluate_model(xTrain=x_train,yTrain=y_train,xTest=x_test,yTest=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))

            bestModel_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[bestModel_name]

            save_object(
                file_path=self.model_Trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_squaree=r2_score(y_test,predicted)

            if best_model_score<0.6:
                raise CustomerException("your models are trash ")
            logging.info("best found model on both training and testing dataset")

            return r2_squaree
        except Exception as e :
            pass
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
                "AdaBoost Regress&or": AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor()
            }

            #hyperparameter tuning 
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }




            
            model_report:dict=evaluate_model(xTrain=x_train,yTrain=y_train,xTest=x_test,yTest=y_test,models=models,param=params)
            
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
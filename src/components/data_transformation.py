import sys
from dataclasses import dataclass
import numpy as np
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.dataTransformationConfig=DataTransformationConfig()
    def getDataTransformer_object(self):

        """this function transform data"""
        try:
            num_Features=["writing_score","reading_score"]
            cat_features=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            categorical_Pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    
                ]
            )

            logging.info(f"Categorical Feautures :{cat_features}")            
            logging.info(f"Numerical features : {num_Features}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_Features),
                    ("cat_pipemine",categorical_Pipeline,cat_features)
                ]

            )
            logging.info("Thedata is preprocessed")

            return preprocessor
           
        except Exception as e:
            raise CustomerException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("The Train and test have been read.")
            logging.info("obtaining preprocessing object.")
            
            preprocessing_obj=self.getDataTransformer_object()
            target_column_name="math_score"

            input_feat_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feat_train_df=train_df[target_column_name]

            input_feat_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feat_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing DataFrame.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feat_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feat_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feat_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feat_test_df)
            ]

            logging.info("saved preprocessing object .")

            save_object(
                file_path=self.dataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr,test_arr#,self.dataTransformationConfig.preprocessor_obj_file_path
        except Exception as e:
            raise CustomerException(e,sys)
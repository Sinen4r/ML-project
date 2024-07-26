import os
import sys
from src.exception import CustomerException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    
    def intitiate_data_ingestion(self):
        logging.info("entred the data ingestion method or component")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info('read the dataset as dataFrame')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("rain test split initaited")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=21)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of the data is completed ")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomerException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.intitiate_data_ingestion()
    DT=DataTransformation()
    DT.initiate_data_transformation(train_data,test_data)
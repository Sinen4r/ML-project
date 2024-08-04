import sys
import pandas as pd
from src.exception import CustomerException
from src.utils import load_objects


class predictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path='artifacts\modell.pkl'
            preprocesspr_path='artifacts\preprocessor.pkl'
            model=load_objects(file_path=model_path)
            preprocessor=load_objects(file_path=preprocesspr_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            CustomerException(e,sys)

class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch:str,
        test_preparation_course:str,
        reading_score,
        writing_score
                 ):
        
        #assigning the varialbles
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def getData_as_DataFrame(self):
        try:
            CDInputDict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(CDInputDict)
        except Exception as e:
            raise CustomerException(e,sys) 
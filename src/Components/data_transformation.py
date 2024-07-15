import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import os
from src.exception import CustomException
from src.logger import logging
from src.Utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        
    def get_data_transformer_obj(self):
        
        '''
        This function is responsible to transform the data 
        '''
        try:
            numerical_colu=["writing_score","reading_score"]
            categorical_colu=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]
            num_pipline=Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy="median")),
                   ("scaler",StandardScaler(with_mean=False))
               ] 
            )
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical columns: {numerical_colu}")
            logging.info(f"Categorical columns:{categorical_colu}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipline,numerical_colu),
                    ("cat_pipeline",cat_pipeline,categorical_colu)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)
            
            logging.info("Reading of  the data is Completed")
            
            logging.info("Obtaining Processing object")
            
            preprocessing_obj=self.get_data_transformer_obj()
            target_col_name="math_score"
            numerical_col=["writing_score","reading_score"]
            
            input_feature_train=train.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train[target_col_name]
            
            input_feature_test=test.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test[target_col_name]
            
            logging.info(f"Applying preprocessing object on traning dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
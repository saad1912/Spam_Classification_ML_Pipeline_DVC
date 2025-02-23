import pandas as pd
import logging
import os
import yaml
from sklearn.model_selection import train_test_split

#Logging

#Ensuring Logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#Logging in Console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


#Logging in File
log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')


#Logging Format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#Adding both types of logging
logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(data_url : str ) -> pd.DataFrame:
    '''Loading Data'''
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data is loaded from url %s ",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Not able to parse the url %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading csv file : %s", e)
        raise
        
    
def preprocess_data(df : pd.DataFrame)->pd.DataFrame:
    '''Pre processing Data'''
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug("Preprocessed data successfully")
    except KeyError as e:
        logger.error("Missing columns in the dataframe : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error during pre-processing : %s",e)
        raise


def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path : str)-> None:
    '''Saving Train and Test Dataframes into csv'''
    
    try:
        raw_data_path = os.path.joins(data_path,'raw')
        train_data.to_csv(os.path.joins(raw_data_path,"train.csv"), index=False)
        test_data.to_csv(os.path.joins(raw_data_path,"test.csv"), index=False)
        logger.debug("Train and Test dataframes saved successfully")
    except Exception as e:
        logger.error("Unexpected error occured while saving train and test dataframes : %s" ,e)
        raise
    
def main():
    test_size = 0.2
    data_path = ""
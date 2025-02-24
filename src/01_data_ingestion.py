import pandas as pd
import logging
import os
import yaml
from sklearn.model_selection import train_test_split

#Logging

#Ensuring Logs directory exists
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger = logging.getLogger('01_data_ingestion')
logger.setLevel('DEBUG')

#Logging in Console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


#Logging in File
log_file_path = os.path.join(log_dir,'01_data_ingestion.log')
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
        return df
    except KeyError as e:
        logger.error("Missing columns in the dataframe : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error during pre-processing : %s",e)
        raise


def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path : str)-> None:
    '''Saving Train and Test Dataframes into csv'''
    
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"), index=False)
        logger.debug("Train and Test dataframes saved successfully")
    except Exception as e:
        logger.error("Unexpected error occured while saving train and test dataframes : %s" ,e)
        raise
    
def main():
    try : 
        test_size = 0.1
        data_path = "https://raw.githubusercontent.com/saad1912/Dataset/refs/heads/main/spam.csv"
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
        logger.debug("Data Ingestion process done successfully")
    except Exception as e:
        logger.error("Unexpected error in completing Data Ingestion process : %s",e)
        print(f"Error : {e}")

if __name__ == '__main__':
    main()
    
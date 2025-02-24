import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('04_model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, '04_model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url : str ) -> pd.DataFrame:
    '''Loading Data'''
    try:
        df = pd.read_csv(data_url)
        df.fillna('', inplace=True)
        logger.debug("Data is loaded from url %s ",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Not able to parse the url %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File Not Found : %s",e)
    except Exception as e:
        logger.error("Unexpected error occured while loading csv file : %s", e)
        raise

def train_model(X_train : np.ndarray, y_train :np.ndarray, params : dict)-> RandomForestClassifier:
    "Train the Model and return the object of trained model"
    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError(f"Mismatch in data length: X_train={X_train.shape[0]}, y_train={y_train.shape[0]}")
        
        logger.debug("Initializing Random Forest model with parameters : %s",params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug("Fitting Model with given parameters")
        clf.fit(X_train,y_train)
        logger.debug("Model Training Complete!!!")
        
        return clf
    
    except ValueError as e:
        logger.error("Value error during model training : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error during model training : %s",e)
        raise
    
    
def save_model(model, file_path : str)-> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise
        
def main():
    try: 
        logger.debug("Loading Parameters from params.yaml...")
        params = load_params('params.yaml')['04_model_training']
        logger.debug("Parameters Loaded Successfully....")
        
        logger.debug("Loading Training Data.......")
        train_data = load_data('./data/final/train_TfIdf.csv')
        logger.debug("Training Data Loaded Successfully....")
        
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        logger.debug("Model Training Started.....")
        clf = train_model(X_train,y_train,params)
        logger.debug("Model Training Completed Successfully...")
        
        logger.debug("Saving Model...")
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
        logger.debug("Model saved successfully....")
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
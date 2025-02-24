import os
import pickle
import pandas as pd
import logging
import yaml
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('05_model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, '05_model_evaluation.log')
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
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading csv file : %s", e)
        raise

def load_model(file_path:str):
    """Load the trained model from a file."""
    try: 
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model Loaded Successfully...")
        return model
    except FileNotFoundError as e:
        logger.error("File Not Found error : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error during loading model from file path")
        raise
        
def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, clf)->dict:
    try: 
        logger.debug("Prediction Process Started....")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        logger.debug("Prediction Process Ended....")
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)
        
        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug("Metrics Calculated Successfully...")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected Error in metric calculation")
        raise
        
        
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try: 
        # Ensure directory exists (handle case where file_path has no directory)
        dir_path = os.path.dirname(file_path)
        if dir_path:  # Avoid making an empty directory
            os.makedirs(dir_path, exist_ok=True)

        # Save JSON file
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics saved to %s", file_path)
    
    except Exception as e:
        logger.error("Error occurred while saving the metrics: %s", e)
        raise
      
        
def main():
    
    try:
        params = load_params('params.yaml')
        test_data = load_data('./data/final/test_TfIdf.csv')
        clf = load_model('models/model.pkl')
        
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        
        
        metrics = evaluate_model(X_test,y_test,clf)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])
            live.log_metric('auc', metrics['auc'])
            live.log_params(params)
        
            
            
        
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

    
if __name__ == '__main__':
    main()
    
    
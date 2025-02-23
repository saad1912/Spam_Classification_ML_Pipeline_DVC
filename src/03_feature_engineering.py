from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pandas as pd
import os


#Logging

#Ensuring Logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger = logging.getLogger('03_feature_engineering')
logger.setLevel('DEBUG')

#Logging in Console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


#Logging in File
log_file_path = os.path.join(log_dir,'03_feature_engineering.log')
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
        df.fillna('', inplace=True)
        logger.debug("Data is loaded from url %s ",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Not able to parse the url %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading csv file : %s", e)
        raise
        

def apply_tfidf(train_data : pd.DataFrame, test_data : pd.DataFrame, max_features : int):
    try: 
        
        tfidf = TfidfVectorizer(max_features = max_features)
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        
        X_test = test_data['text'].values
        y_test = test_data['target'].values
        
        
        
        X_train_bow = tfidf.fit_transform(X_train)
        X_test_bow = tfidf.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['target'] = y_test
        logger.debug("Vectorization of Feature is complete!")
        
        return train_df, test_df
    except Exception as e:
        logger.error("Unexpected error during vectorization : %s",e)
        raise
    
    
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
   

def main():
    try :
        train_data = load_data('./data/interim/preprocessed_train.csv')
        test_data = load_data('./data/interim/preprocessed_test.csv')
        
        train_df,test_df = apply_tfidf(train_data,test_data,max_features=500)
        save_data(train_df, os.path.join("./data","final","train_TfIdf.csv"))
        save_data(test_df, os.path.join("./data","final","test_TfIdf.csv"))
        logger.debug("Feature Engineering Stage Completed Successfully!")
    
    except Exception as e: 
        logger.error("Unexpected error during Feature Engineering Stage : %s",e)
        

if __name__=='__main__':
    main()
              
    
    
    
    
    
    
    
    
    
   
     

    
    
    
    
    
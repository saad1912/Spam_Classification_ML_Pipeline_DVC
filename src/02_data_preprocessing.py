import pandas as pd
import logging
import os
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



#Logging

#Ensuring Logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger = logging.getLogger('02_data_preprocessing')
logger.setLevel('DEBUG')

#Logging in Console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


#Logging in File
log_file_path = os.path.join(log_dir,'02_data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')


#Logging Format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#Adding both types of logging
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    # Convert text to lowercase
    ps = PorterStemmer()
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)
    
    # Removing special characters (keeping only alphanumeric words)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # Removing stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming (reducing words to root form)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    # Join words back into a single string
    return " ".join(y)


def pre_process(df:pd.DataFrame, text_column='text', target_column = 'target')->pd.DataFrame:
    
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    
    try:
    #Label Encoding Target Column
        lb_enc = LabelEncoder()
        df[target_column] = lb_enc.fit_transform(df[target_column])
        logger.debug("Target Column Encoded")

        
        #Dropping duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicate Rows Dropped")
        
        
        #Transforming Text Column through Stemming
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        logger.debug("Successfully Perfomed Preprocessing of Dataset")
        return df
    
    except KeyError as e:
        logger.error("Column Not Found error : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error in the Preprocessing step of pipeline : %s",e)
        raise
    

def main(text_column = 'text',target_column = 'target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    
    try : 
        #Loading Train and Test Data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Train and Test Data loaded successfully")
        
        #Pre_processing Train and Test Data
        logger.debug("Train Data Transformation Started...")
        preprocessed_train_df = pre_process(train_data,text_column,target_column)
        logger.debug("Train Data Transformation Ended...")
        
        logger.debug("Test Data Transformation Started...")
        preprocessed_test_df = pre_process(test_data,text_column,target_column)
        logger.debug("Test Data Transformation Ended...")
        
        logger.debug("Train and Test data preprocessed successfully")
        
        interim_data_path = os.path.join("./data","interim")
        os.makedirs(interim_data_path, exist_ok=True)
        
        
        preprocessed_train_df.to_csv(os.path.join(interim_data_path,"preprocessed_train.csv"), index=False)
        preprocessed_test_df.to_csv(os.path.join(interim_data_path,"preprocessed_test.csv"), index=False)
        
        
        logger.debug("Preprocessed data saved to : %s", interim_data_path)
        
    except FileNotFoundError as e:
        logger.error("File Not Found Error : %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No Data error : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error in the Data Preprocessing Stage")
    

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
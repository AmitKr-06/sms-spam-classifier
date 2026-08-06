import sys
import os
import re
from dataclasses import dataclass

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_config

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english')) | {'u', 'ur', 'im', 'dont', 'didnt', 'cant'}
STEMMER = PorterStemmer()
SPAM_WORDS = ['call', 'txt', 'free', 'win', 'cash', 'prize', 'claim',
              'guarante', 'urgent', 'repli', 'stop', 'text', 'send']

FEATURE_COLS = ['sms_stemmed', 'word_count', 'unique_words', 'punctuation_count',
                 'capitalized_words', 'has_numbers', 'has_url', 'has_exclamation', 'spam_word_count']
META_COLS = FEATURE_COLS[1:]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def full_text_pipeline(text):
    tokens = clean_text(text).split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(STEMMER.stem(w) for w in tokens)


def engineer_features(df):
    df = df.copy()
    df['word_count'] = df['sms'].str.split().str.len()
    df['unique_words'] = df['sms'].apply(lambda x: len(set(x.split())))
    df['punctuation_count'] = df['sms'].apply(lambda x: len(re.findall(r'[!?.]', x)))
    df['capitalized_words'] = df['sms'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['has_numbers'] = df['sms'].str.contains(r'\d').astype(int)
    df['has_url'] = df['sms'].str.contains(r'http|www|\.com', case=False).astype(int)
    df['has_exclamation'] = df['sms'].str.contains('!').astype(int)
    df['sms_stemmed'] = df['sms'].apply(full_text_pipeline)
    df['spam_word_count'] = df['sms_stemmed'].apply(
        lambda x: sum(1 for w in SPAM_WORDS if w in x.split()))
    return df


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.tfidf_params = load_config().get("tfidf", {})

    def get_data_transformer_object(self):
        try:
            max_features = self.tfidf_params.get("max_features", 3000)
            preprocessor = ColumnTransformer([
                ('tfidf', TfidfVectorizer(max_features=max_features), 'sms_stemmed'),
                ('scaler', MinMaxScaler(), META_COLS)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            train_df = engineer_features(train_df)
            test_df = engineer_features(test_df)

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "label"

            X_train = preprocessing_obj.fit_transform(train_df[FEATURE_COLS])
            y_train = train_df[target_column_name]
            X_test = preprocessing_obj.transform(test_df[FEATURE_COLS])
            y_test = test_df[target_column_name]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
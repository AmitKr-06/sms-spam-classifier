import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import load_config


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw", "sms.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.params = load_config().get("train", {})

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            if os.path.exists(self.ingestion_config.raw_data_path):
                logging.info("Local raw file found at data/raw/sms.csv, reading it")
                df = pd.read_csv(self.ingestion_config.raw_data_path)
            else:
                logging.info("No local raw file found, downloading SMS Spam dataset from HuggingFace")
                from datasets import load_dataset
                dataset = load_dataset("sms_spam")
                df = dataset["train"].to_pandas()
                df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
                logging.info(f"Dataset downloaded and cached, shape: {df.shape}")

            # Deduplicate regardless of source (local file or fresh download) to
            # prevent data leakage between train/test splits.
            before = len(df)
            df = df.drop_duplicates(subset=["sms"])
            removed = before - len(df)
            logging.info(f"Removed {removed} duplicate rows ({removed/before*100:.1f}%). Shape now: {df.shape}")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df,
                test_size=self.params.get("test_size", 0.2),
                random_state=self.params.get("random_state", 42),
                stratify=df["label"],
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
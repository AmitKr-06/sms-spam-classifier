import os
import sys
import pickle
import yaml

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_config(config_path: str = os.path.join("config", "config.yaml")) -> dict:
    """
    Loads config/config.yaml. Called from anywhere in the pipeline so
    hyperparameters/paths aren't hardcoded inside components.
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)


def load_schema(schema_path: str = os.path.join("config", "schema.yaml")) -> dict:
    """
    Loads config/schema.yaml. Used by data_validation.py to check the
    dataset's columns/dtypes/allowed values against an explicit spec
    instead of constants hardcoded in the validation code.
    """
    try:
        with open(schema_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)
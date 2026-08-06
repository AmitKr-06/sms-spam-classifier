import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import engineer_features, FEATURE_COLS


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, raw_message: str):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data = CustomData(raw_message).get_data_as_data_frame()
            data_scaled = preprocessor.transform(data[FEATURE_COLS])
            preds = model.predict(data_scaled)
            proba = model.predict_proba(data_scaled)
            return preds, proba
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, sms: str):
        self.sms = sms

    def get_data_as_data_frame(self):
        try:
            df = pd.DataFrame({"sms": [self.sms]})
            return engineer_features(df)
        except Exception as e:
            raise CustomException(e, sys)
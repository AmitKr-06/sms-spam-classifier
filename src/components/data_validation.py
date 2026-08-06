import os
import sys
import json
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging

REQUIRED_COLUMNS = ["sms", "label"]
VALID_LABELS = {0, 1}


@dataclass
class DataValidationConfig:
    validation_report_path: str = os.path.join("artifacts", "validation_report.json")


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_dataframe(self, df: pd.DataFrame, name: str = "dataset") -> dict:
        try:
            issues = []

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
                # If columns are missing outright, the rest of the checks can't run safely.
                report = {"name": name, "status": "failed", "issues": issues}
                logging.warning(f"Validation failed for {name}: {issues}")
                return report

            if df.empty:
                issues.append("Dataset is empty")

            null_sms = int(df["sms"].isnull().sum())
            if null_sms > 0:
                issues.append(f"{null_sms} null values in 'sms' column")

            empty_sms = int((df["sms"].astype(str).str.strip() == "").sum())
            if empty_sms > 0:
                issues.append(f"{empty_sms} empty strings in 'sms' column")

            dup_count = int(df.duplicated().sum())
            if dup_count > 0:
                issues.append(f"{dup_count} duplicate rows found")

            null_labels = int(df["label"].isnull().sum())
            if null_labels > 0:
                issues.append(f"{null_labels} null values in 'label' column")

            unexpected_labels = set(df["label"].dropna().unique()) - VALID_LABELS
            if unexpected_labels:
                issues.append(f"Unexpected label values: {unexpected_labels}")

            if not pd.api.types.is_object_dtype(df["sms"]) and not pd.api.types.is_string_dtype(df["sms"]):
                issues.append("'sms' column is not string/object dtype")

            status = "failed" if issues else "passed"
            if issues:
                logging.warning(f"Validation issues in {name}: {issues}")
            else:
                logging.info(f"Validation passed for {name}")

            return {"name": name, "status": status, "issues": issues}
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_report = self.validate_dataframe(train_df, name="train")
            test_report = self.validate_dataframe(test_df, name="test")

            os.makedirs(os.path.dirname(self.validation_config.validation_report_path), exist_ok=True)
            with open(self.validation_config.validation_report_path, "w") as f:
                json.dump({"train": train_report, "test": test_report}, f, indent=4)

            if train_report["status"] == "failed" or test_report["status"] == "failed":
                logging.warning("Data validation found issues. Review artifacts/validation_report.json")

            return train_report, test_report
        except Exception as e:
            raise CustomException(e, sys)
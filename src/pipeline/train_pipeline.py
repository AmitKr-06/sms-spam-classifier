from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    validation = DataValidation()
    train_report, test_report = validation.initiate_data_validation(train_path, test_path)
    logging.info(f"Train validation report: {train_report}")
    logging.info(f"Test validation report: {test_report}")

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    best_name, best_score = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    print(f"Training complete. Best model: {best_name}, F1 score: {best_score:.4f}")

    evaluator = ModelEvaluation()
    metrics = evaluator.initiate_model_evaluation(X_test, y_test)
    print(f"Evaluation metrics: {metrics}")
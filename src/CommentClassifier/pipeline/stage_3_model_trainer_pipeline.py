from src.CommentClassifier.config.configuration import ConfigurationManager
from src.CommentClassifier.components.model_trainer import ModelTrainer
from src.CommentClassifier.logging import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        # config manager
        config=ConfigurationManager()
        # Run Model Training
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        df = model_trainer.load_dataset()
        X_train, X_test, y_train, y_test, vectorizer = model_trainer.preprocess_data(df)

        model = model_trainer.train_model(X_train, y_train)
        model_trainer.save_model(model, vectorizer)
        logger.info("Model training completed successfully!")


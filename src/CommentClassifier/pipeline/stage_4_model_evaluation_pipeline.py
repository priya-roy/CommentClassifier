from src.CommentClassifier.config.configuration import ConfigurationManager
from src.CommentClassifier.components.model_evaluation import ModelEvaluation
from src.CommentClassifier.logging import logger

class modelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        # config manager
        config = ConfigurationManager()
        # Run Model Evaluation
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(model_evaluation_config)
        model_evaluation.run_evaluation()
        logger.info("Model evaluation completed successfully!")
        
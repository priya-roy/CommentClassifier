from src.CommentClassifier.config.configuration import ConfigurationManager
from src.CommentClassifier.components.data_transformation import DataTransformation
from src.CommentClassifier.components.eda import EDA
from src.CommentClassifier.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        # config manager
        config=ConfigurationManager()
        # Run Data Transformation
        data_transformation_config=config.get_data_transformation_config()
        transformer = DataTransformation(config=data_transformation_config)
        transformer.transform_data()
        # Run EDA
        data_transformation_config = config.get_data_transformation_config()
        eda = EDA(config=data_transformation_config)  
        eda.run_eda()
        
        
        
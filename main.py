from src.CommentClassifier.logging import logger
from src.CommentClassifier.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.CommentClassifier.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline


STAGE_NAME="Data Ingestion stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Data Transformation stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_transform_pipeline=DataTransformationTrainingPipeline()
    data_transform_pipeline.initiate_data_transformation()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e

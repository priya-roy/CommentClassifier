from src.CommentClassifier.constants import *
from src.CommentClassifier.utils.common import read_yaml, create_directories

from src.CommentClassifier.entity import DataIngestionConfig,DataTransformationConfig

class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config=read_yaml(config_path)
        self.paramss=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )

        return data_ingestion_config
    
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config=self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
            text_column=config.text_column,
            label_column=config.label_column,
            stopwords=config.stopwords,
            lowercase=config.lowercase,
            remove_special_characters=config.remove_special_characters,
            remove_extra_spaces=config.remove_extra_spaces
        )

        return data_transformation_config
    
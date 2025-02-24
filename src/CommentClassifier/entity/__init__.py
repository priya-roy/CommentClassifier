from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
    text_column: str
    label_column: bool
    stopwords: str
    lowercase: bool
    remove_special_characters: bool
    remove_extra_spaces: bool

@dataclass    
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    text_column: str
    label_column: bool
    penalty: str
    random_state: int
    max_iter: int
    class_weight: str
    max_features: str
    ngram_range: list
    stop_words: str
    min_df: int
    max_df:float
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    metrics: list[str]
    validation_split: float
    cross_validation: bool
    cv_folds: int
    text_column: str
    label_column: bool

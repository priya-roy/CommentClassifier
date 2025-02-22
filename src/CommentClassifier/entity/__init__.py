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
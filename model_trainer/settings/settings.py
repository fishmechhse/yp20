from pydantic import Field

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_dir: str = Field("./", alias='MODEL_DIR')
    num_cores: int = Field(3, alias='NUM_CORES')
    max_models_loaded: int = Field(3, alias='MAX_MODELS_LOADED')

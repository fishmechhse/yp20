from pydantic import BaseModel
from typing import List, Union, Optional
from enum import Enum
class MLModel(BaseModel):
    id: str
    type: str

class StatusResponse(BaseModel):
    status: str

class ValidationError(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: Optional[List[ValidationError]]

class ModelType(str, Enum):
    SVC = 'svc'
    LOGIC = 'logistic'
    UNDEFINED = 'undefined'

class ModelConfig(BaseModel):
    id: str
    ml_model_type: ModelType
    hyperparameters: dict

class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[str]
    config: ModelConfig

class FitResponse(BaseModel):
    message: str

class UnloadRequest(BaseModel):
    id: str
class LoadRequest(BaseModel):
    id: str

class LoadResponse(BaseModel):
    message: str

class ModelListResponse(BaseModel):
    models: List[MLModel]

class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]

class RemoveResponse(BaseModel):
    message: str


class UnloadResponse(BaseModel):
    message: str

from typing import List, Union, Optional, Annotated
from enum import Enum
from pydantic import BaseModel, Field


class MLModel(BaseModel):
    id: Annotated[str, Field(description="Model ID")]
    type: Annotated[str, Field(description="Model type")]


class StatusResponse(BaseModel):
    status: Annotated[str, Field(description="Model status")]


class ValidationError(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class HTTPValidationError(BaseModel):
    detail: Optional[List[ValidationError]]


class ModelType(str, Enum):
    SVC = 'svc'  # SVC model
    LOGIC = 'logistic'  # LogisticRegression model
    UNDEFINED = 'undefined'  # undefined model


class ModelConfig(BaseModel):
    id: Annotated[str, Field(description="Model ID")]
    ml_model_type: (
        Annotated)[ModelType, Field(description="model type. logistic or svc")]
    hyperparameters: (
        Annotated)[dict, Field(description="hyperparameters for model")]


class FitRequest(BaseModel):
    X: Annotated[List[List[float]], Field(description="Input features")]
    y: Annotated[List[str], Field(description="Target labels")]
    config: Annotated[ModelConfig, Field(description="Configuration settings")]


class FitResponse(BaseModel):
    message: Annotated[str, Field(description="Status model after training")]


class UnloadRequest(BaseModel):
    id: Annotated[str, Field(description="Model ID for unloading")]


class LoadRequest(BaseModel):
    id: Annotated[str, Field(description="Model ID for unloading")]


class LoadResponse(BaseModel):
    message: Annotated[str, Field(description="Model status")]


class ModelListResponse(BaseModel):
    models: List[MLModel]


class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]


class PredictionModel(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionModel]


class RemoveResponse(BaseModel):
    message: str


class UnloadResponse(BaseModel):
    message: str

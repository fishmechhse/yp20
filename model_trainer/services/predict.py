from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from typing import List

from model_trainer.models.models import ModelType, FitRequest


def fit_logistic_regression(x: List[List[float]], y: List[str], hyperparameters: dict) -> Pipeline:
    # todo: check hyperparameters
    default_attr = {
        "solver": "liblinear",
        "max_iter": 600,
        "class_weight": "balanced"
    }

    if len(hyperparameters) == 0:
        hyperparameters = default_attr

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(**hyperparameters))
    ])
    pipeline.fit(x, y)
    return pipeline


def fit_svc(x: List[List[float]], y: List[str], hyperparameters: dict) -> Pipeline:
    # todo:  check hyperparameters
    default_attr = {
        "class_weight": "balanced"
    }

    if len(hyperparameters) == 0:
        hyperparameters = default_attr

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('logistic', SVC(**hyperparameters))
    ])
    pipeline.fit(x, y)

    return pipeline


class ModelManager:
    __dict_model = dict()

    def __init__(self):
        self.__dict_model = dict()

    def fit_request(self, set: List[FitRequest]):
        pass

    def clear(self):
        self.__dict_model = dict()


class TrainedModel:
    __regressor = None
    __id: str
    __model_type: ModelType

    def __init__(self, id: str, model_type: ModelType, regressor):
        self.__regressor = regressor
        self.__id = id
        self.__model_type = model_type

    def get_regressor(self):
        return self.__regressor

    def get_id(self):
        return self.__id

    def get_model_type(self):
        return self.__model_type


def fit_model(fit_req: FitRequest, return_dict):
    match fit_req.config.ml_model_type:
        case ModelType.SVC:
            return_dict[fit_req.config.id] = TrainedModel(
                id=fit_req.config.id,
                regressor=fit_svc(fit_req.X, fit_req.y, fit_req.config.hyperparameters),
                model_type=ModelType.SVC
            )
        case ModelType.LOGIC:
            return_dict[fit_req.config.id] = TrainedModel(
                id=fit_req.config.id,
                regressor=fit_logistic_regression(fit_req.X, fit_req.y, fit_req.config.hyperparameters),
                model_type=ModelType.LOGIC
            )
        case _:
            raise Exception("undefined model type")


def fit_model_dataframe(fit_req: FitRequest, return_dict):
    match fit_req.config.ml_model_type:
        case ModelType.SVC:
            return_dict[fit_req.config.id] = TrainedModel(
                id=fit_req.config.id,
                regressor=fit_svc(fit_req.X, fit_req.y, fit_req.config.hyperparameters),
                model_type=ModelType.SVC
            )
        case ModelType.LOGIC:
            return_dict[fit_req.config.id] = TrainedModel(
                id=fit_req.config.id,
                regressor=fit_logistic_regression(fit_req.X, fit_req.y, fit_req.config.hyperparameters),
                model_type=ModelType.LOGIC
            )
        case _:
            raise Exception("undefined model type")

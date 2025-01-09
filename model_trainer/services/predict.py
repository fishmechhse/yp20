from time import sleep

from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import List


from model_trainer.models.models import ModelType, FitRequest


def fit_logistic_regression(x: List[List[float]], y: List[float], hyperparameters: dict) -> LogisticRegression:
    # todo: check hyperparameters
    reg = LogisticRegression(**hyperparameters)
    reg.fit(x, y)
    return reg


def fit_linear_regression(x: List[List[float]], y: List[float], hyperparameters: dict) -> LinearRegression:
    # todo:  check hyperparameters
    reg = LinearRegression(**hyperparameters)
    reg.fit(x, y)
    return reg


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
        case ModelType.LINEAR:
            return_dict[fit_req.config.id] = TrainedModel(
                id=fit_req.config.id,
                regressor=fit_linear_regression(fit_req.X, fit_req.y, fit_req.config.hyperparameters),
                model_type=ModelType.LINEAR
            )
            if "_60_" in fit_req.config.id:
                print(f'{fit_req.config.id} fitting model during 60 seconds...')  # or logger.debug(), logger.error(), etc.
                sleep(60)
        case ModelType.LOGIC:
            return_dict[fit_req.config.id] = TrainedModel(
                id=fit_req.config.id,
                regressor=fit_logistic_regression(fit_req.X, fit_req.y, fit_req.config.hyperparameters),
                model_type=ModelType.LOGIC
            )
        case _:
            raise Exception("undefined model type")



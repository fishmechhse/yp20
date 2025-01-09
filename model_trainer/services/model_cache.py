from multiprocessing import Lock

from model_trainer.services.predict import TrainedModel


class ModelRegistry:
    __model_set: dict
    __max_size: int
    __locker: Lock

    def __init__(self, max_size: int):
        self.__model_set = dict()
        self.__max_size = max_size
        self.__locker = Lock()

    def remove_model(self, model_id: str):
        self.__locker.acquire()
        try:
            del self.__model_set[model_id]
        finally:
            self.__locker.release()

    def get_model(self, model_id: str):
        self.__locker.acquire()
        try:
            exists = (model_id in self.__model_set)
            if not exists:
                return None
            model = self.__model_set[model_id]
            return model
        finally:
            self.__locker.release()

    def get_models_id_list(self) -> []:
        self.__locker.acquire()
        try:
            res = []
            for key in self.__model_set:
                res.append(key)
            return res
        finally:
            self.__locker.release()

    def add_model(self, trained_model: TrainedModel) -> bool:
        self.__locker.acquire()
        try:
            key = trained_model.get_id()
            exists = key in self.__model_set

            if exists:
                self.__model_set[trained_model.get_id()] = trained_model
                return True

            if (not exists) and (len(self.__model_set) < self.__max_size):
                self.__model_set[trained_model.get_id()] = trained_model
                return True

            return False
        finally:
            self.__locker.release()


class NotFoundModel:
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class LimitLoadedModelsException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

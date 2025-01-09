import joblib
from pathlib import Path
from multiprocessing import Lock
import json
from sklearn.linear_model import LogisticRegression, LinearRegression

from model_trainer.models.models import ModelType
from model_trainer.services.predict import TrainedModel


def make_full_name(directory, filename: str) -> str:
    return f'{directory}/{filename}'


class ModelStorage:
    __directory: str
    __locker: Lock
    __index_file = 'index.json'
    __ext = '.pkl'
    __model_dict = dict()

    def __init__(self, directory: str):
        self.__directory = directory
        self.__model_dict = dict()
        self.__locker = Lock()

        self.__load_index()

    def remove_models(self) -> []:
        self.__locker.acquire()
        try:
            res = []
            for model_id in self.__model_dict.copy():
                removed_model_description = self.__pop_model(model_id)
                res.append(removed_model_description)
            return res
        finally:
            self.__save_index()
            self.__locker.release()

    def remove_model(self, model_id: str):
        self.__locker.acquire()
        try:
            m = self.__pop_model(model_id)
            return m
        finally:
            self.__save_index()
            self.__locker.release()

    def __pop_model(self, model_id: str):
        try:
            model_description = self.__get_model_description(model_id)
            if model_description is None:
                raise NotFoundModelException(message=f"not found model '{model_id}'")
            pf = Path(make_full_name(self.__directory, model_id + self.__ext))
            if not pf.is_file():
                raise NotFoundModelException(message=f"not found file of model'{model_id}'")

            pf.unlink()
            self.__remove_from_index(model_id)
            return model_description
        except FileNotFoundError as ef:
            raise NotFoundModelException(str(ef))

    def __get_model_description(self, model_id: str):
        exists = model_id in self.__model_dict
        if not exists:
            return None
        el = self.__model_dict[model_id]
        return {
            'id': el['id'],
            'type': self.__convert_type(el['type']),
        }

    def __remove_from_index(self, model_id: str):
        del self.__model_dict[model_id]

    def __load_index(self):
        index_file = make_full_name(self.__directory, self.__index_file)
        pf = Path(index_file)
        if pf.is_file():
            self.__model_dict = json.loads(pf.read_text())

    def __convert_type(self, mt: ModelType) -> str:
        match mt:
            case ModelType.LINEAR:
                return 'linear'
            case ModelType.LOGIC:
                return 'logic'
        return 'undefined'

    def get_trained_model_id_with_types(self) -> []:
        res = []
        for key in self.__model_dict:
            set = self.__model_dict[key]
            res.append({
                'id': set['id'],
                'type': self.__convert_type(set['type']),
            })
        return res

    def get_type_for_regressor(self, regressor) -> ModelType:
        if isinstance(regressor, LinearRegression):
            return ModelType.LINEAR
        if isinstance(regressor, LogisticRegression):
            return ModelType.LOGIC
        return ModelType.UNDEFINED

    def load_regressor(self, model_id: str) -> TrainedModel:
        m_id = model_id + ".pkl"
        my_file = Path(make_full_name(self.__directory, m_id))
        if not my_file.is_file():
            raise NotFoundModelException(message=f"not found file of model '{model_id}'")
        try:
            f = open(make_full_name(self.__directory, m_id), "rb")
            clf = joblib.load(f)
            tm = TrainedModel(id=model_id, regressor=clf, model_type=self.get_type_for_regressor(clf))
            return tm
        except Exception as e:
            raise UnprocessableException(str(e))

    def save_regressor(self, trained_model: TrainedModel):
        self.__locker.acquire()
        try:
            filename = trained_model.get_id() + ".pkl"
            filename_with_dir = make_full_name(self.__directory, filename)
            with open(filename_with_dir, 'wb') as f:
                joblib.dump(trained_model.get_regressor(), f)

            self.__model_dict[trained_model.get_id()] = {
                'type': trained_model.get_model_type(),
                'id': trained_model.get_id()
            }

            self.__save_index()
        except Exception as e:
            raise UnprocessableException(str(e))
        finally:
            self.__locker.release()

    def __save_index(self):
        index_file = make_full_name(self.__directory, self.__index_file)
        with open(index_file, 'w') as fp:
            json.dump(self.__model_dict, fp)


class NotFoundModelException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnprocessableException(Exception):
    def __init__(self, message):
        super().__init__(message)

from fastapi import HTTPException
from fastapi import APIRouter
from http import HTTPStatus
from typing import List


from model_trainer.models.models import FitRequest, FitResponse, LoadResponse, LoadRequest, UnloadResponse, \
    UnloadRequest, StatusResponse, PredictionResponse, PredictRequest, ModelListResponse, MLModel, RemoveResponse
from model_trainer.services.model_cache import LimitLoadedModelsException, ModelRegistry
from model_trainer.services.model_storage import ModelStorage, NotFoundModelException
from model_trainer.services.predict import fit_model, TrainedModel

import multiprocessing

from model_trainer.services.shared_counter import SharedCounter
from model_trainer.settings.settings import Settings

import asyncio

router = APIRouter()


def get_required_number_workers(items: List[FitRequest]) -> int:
    return len(items)


env_settings = Settings()

shared_counter = SharedCounter(env_settings.num_cores)
model_storage = ModelStorage(env_settings.model_dir)
model_registry = ModelRegistry(env_settings.max_models_loaded)


# API endpoints
@router.post("/fit", status_code=HTTPStatus.OK, response_model=List[FitResponse])
async def fit(req: List[FitRequest]):
    req_workers_num = get_required_number_workers(req)
    num_vacant_workers = shared_counter.try_lock_workers(req_workers_num)

    if num_vacant_workers == 0:
        raise HTTPException(status_code=429, detail="Too many active training processes.")
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, async_process, req)
    finally:
        shared_counter.decrement(num_vacant_workers)


def async_process(req: List[FitRequest]):
    res: List[FitResponse] = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for model in req:
        m = model
        processes.append(multiprocessing.Process(target=fit_model, args=(m, return_dict)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for trained_model in return_dict.values():
        if not isinstance(trained_model, TrainedModel):
            raise Exception("model wasn't trained")
        model_res = FitResponse(message=f"Model '{trained_model.get_id()}' trained and saved")
        model_storage.save_regressor(trained_model)
        res.append(model_res)
    return res


@router.post("/load", response_model=List[LoadResponse])
async def load(load_request: LoadRequest):
    res: List[LoadResponse] = []
    try:
        trained_model = model_storage.load_regressor(load_request.id)
        result = model_registry.add_model(trained_model)
        if not result:
            raise LimitLoadedModelsException("limit loaded models")

        res.append(LoadResponse(message=f"Model '{trained_model.get_id()}' loaded"))
        return res
    except Exception as ex:
        raise HTTPException(status_code=422, detail=str(ex))


@router.post("/unload", response_model=List[UnloadResponse])
async def unload(req: UnloadRequest):
    res: List[UnloadResponse] = []
    try:
        model_registry.remove_model(req.id)
        res.append(UnloadResponse(message=f"Model '{req.id}' unloaded"))
        return res
    except Exception as ex:
        raise HTTPException(status_code=422, detail=str(ex))


@router.get("/get_status", response_model=List[StatusResponse])
async def get_status():
    list_ids = model_registry.get_models_id_list()
    res: List[StatusResponse] = []
    for model_id in list_ids:
        res.append(StatusResponse(status=f"{model_id} Status Ready"))
    return res


@router.post("/predict", response_model=List[PredictionResponse])
async def predict(req: List[PredictRequest]):
    res: List[PredictionResponse] = []
    for pred_op in req:
        trained_model = model_registry.get_model(pred_op.id)
        if trained_model is None:
            raise HTTPException(status_code=404,
                                detail=f"model '{pred_op.id}' not found")
        reg = trained_model.get_regressor()
        y = reg.predict(pred_op.X)
        res.append(PredictionResponse(predictions=y))
    return res


@router.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    list_id_types = model_storage.get_trained_model_id_with_types()
    models: List[MLModel] = []
    for v in list_id_types:
        models.append(MLModel(id=v['id'], type=v['type']))
    res = []
    res.append(ModelListResponse(models=models))
    return res


@router.delete("/remove/{model_id}", response_model=List[RemoveResponse])
async def remove_id(model_id: str):
    res: List[RemoveResponse] = []
    try:
        model = model_storage.remove_model(model_id)
        res.append(RemoveResponse(message=f"Model '{model['id']}' removed"))
        return res
    except NotFoundModelException as ex:
        raise HTTPException(status_code=404, detail=str(ex))


@router.delete("/remove_all", response_model=List[RemoveResponse])
async def remove():
    res: List[RemoveResponse] = []
    models = model_storage.remove_models()
    for m in models:
        res.append(RemoveResponse(message=f"Model '{m['id']}' removed"))
    return res


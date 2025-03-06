from aiogram.fsm.state import StatesGroup, State

class TrainStates(StatesGroup):
    waiting_for_model_id = State()
    waiting_for_model_type = State()
    waiting_for_hyperparameters = State()
    waiting_for_features = State()
    waiting_for_labels = State()

class TrainCSVStates(StatesGroup):
    waiting_for_model_id = State()
    waiting_for_model_id_csv = State()
    waiting_for_model_type = State()
    waiting_for_model_type_csv = State()
    waiting_for_csv = State()
    next_step = State()

class PredictStates(StatesGroup):
    waiting_for_model_id = State()
    waiting_for_features = State()

class ModelIDState(StatesGroup):
    waiting_for_model_id = State()

class RemoveAllStates(StatesGroup):
    waiting_for_confirmation = State()

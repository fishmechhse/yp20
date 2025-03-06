import json
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
import aiohttp
import httpx
from io import BytesIO
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from states import TrainStates, TrainCSVStates, PredictStates, ModelIDState, RemoveAllStates
from config import API_URL  # локальная реализация, нужно будет менять

router = Router()

@router.message(Command("start"))
async def start_command(message: types.Message):
    text = (
        "Привет! Я бот для работы с ЭКГ\n"
        "Доступные команды:\n"
        "/train - тренировка модели (ввод данных вручную)\n"
        "/train_csv - тренировка модели через CSV файл\n"
        "/predict - сделать предсказание\n"
        "/list - список моделей\n"
        "/status - статус загруженных моделей\n"
        "/load - загрузить модель\n"
        "/unload - выгрузить модель\n"
        "/remove - удалить модель\n"
        "/remove_all - удалить все модели\n"
    )
    await message.answer(text)

@router.message(Command("train"))
async def cmd_train(message: types.Message, state: FSMContext):
    await message.answer("Введите идентификатор модели:")
    await state.set_state(TrainStates.waiting_for_model_id)

@router.message(TrainStates.waiting_for_model_id)
async def process_model_id(message: types.Message, state: FSMContext):
    await state.update_data(model_id=message.text)
    await state.update_data(model_id=message.text)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="SVC", callback_data="svc")],
        [InlineKeyboardButton(text="Logic", callback_data="logic")]
    ])
    await message.answer("Выберите тип модели:", reply_markup=keyboard)
    await state.set_state(TrainCSVStates.waiting_for_model_type)


@router.callback_query(lambda c: c.data in ["svc", "logic"])
async def process_model_type(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "svc":
        model_type = "svc"
    elif callback_query.data == "logic":
        model_type = "logistic"
    else:
        model_type = "undefined"
    await state.update_data(model_type=model_type)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="По умолчанию", callback_data="hyper_default")],
        [InlineKeyboardButton(text="Ввести вручную", callback_data="hyper_manual")]
    ])
    await callback_query.answer("Выберите способ задания гиперпараметров:", reply_markup=keyboard)

@router.callback_query(lambda c: c.data in ["hyper_default", "hyper_manual"])
async def process_hyper_choice(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "hyper_default":
        await state.update_data(hyperparameters={})
        await callback_query.message.answer(
            "Гиперпараметры установлены по умолчанию.\n"
            "Введите признаки для обучения.\n"
            "Формат: строки, разделённые точкой с запятой, значения через запятую.\n"
            "Пример: 1,2,3; 4,5,6"
        )
        await state.set_state(TrainStates.waiting_for_features)
    elif callback_query.data == "hyper_manual":
        await callback_query.message.answer("Введите гиперпараметры в формате JSON:")
        await state.set_state(TrainStates.waiting_for_hyperparameters)
    await callback_query.answer()

@router.message(TrainStates.waiting_for_hyperparameters)
async def process_hyperparameters(message: types.Message, state: FSMContext):
    text = message.text.strip()
    if text:
        try:
            hyperparams = json.loads(text)
        except json.JSONDecodeError:
            await message.answer("Некорректный формат JSON. Попробуйте ещё раз:")
            return
    else:
        hyperparams = {}
    await state.update_data(hyperparameters=hyperparams)
    await message.answer(
        "Введите признаки для обучения.\n"
        "Формат: строки, разделённые точкой с запятой, значения через запятую.\n"
        "Пример: 1,2,3; 4,5,6"
    )
    await state.set_state(TrainStates.waiting_for_features)

@router.message(TrainStates.waiting_for_features)
async def process_features(message: types.Message, state: FSMContext):
    try:
        rows = message.text.split(";")
        features = [list(map(float, row.split(","))) for row in rows if row.strip()]
    except Exception:
        await message.answer("Ошибка при обработке признаков. Проверьте формат ввода.")
        return
    await state.update_data(features=features)
    await message.answer(
        "Введите метки (labels) для обучения, разделённые запятой.\nПример: class1, class2"
    )
    await state.set_state(TrainStates.waiting_for_labels)

@router.message(TrainStates.waiting_for_labels)
async def process_labels(message: types.Message, state: FSMContext):
    data = await state.get_data()
    labels = [lbl.strip() for lbl in message.text.split(",") if lbl.strip()]
    if len(labels) != len(data.get("features", [])):
        await message.answer("Количество меток должно соответствовать количеству строк признаков. Попробуйте снова:")
        return
    await state.update_data(labels=labels)
    payload = {
        "X": data["features"],
        "y": labels,
        "config": {
            "id": data["model_id"],
            "ml_model_type": data["model_type"],
            "hyperparameters": data["hyperparameters"]
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_URL}/fit", json=payload)
            if response.status_code == 200:
                result = response.json()
                await message.answer(f"Модель обучена: {result[0]['message']}")
            else:
                await message.answer(f"Ошибка: {response.text}")
        except Exception as e:
            await message.answer(f"Ошибка при вызове API: {str(e)}")
    await state.clear()

@router.message(Command("train_csv"))
async def cmd_train_csv(message: types.Message, state: FSMContext):
    await message.answer("Введите идентификатор модели:")
    await state.set_state(TrainCSVStates.waiting_for_model_id_csv)

@router.message(TrainCSVStates.waiting_for_model_id_csv)
async def process_csv_model_id(message: types.Message, state: FSMContext):
    await state.update_data(model_id=message.text)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="SVC", callback_data="svc_csv")],
        [InlineKeyboardButton(text="Logic", callback_data="logic_csv")]
    ])
    await message.answer("Выберите тип модели:", reply_markup=keyboard)
    await state.set_state(TrainCSVStates.waiting_for_model_type_csv)


@router.callback_query(lambda c: c.data in ["svc_csv", "logic_csv"])
async def process_csv_model_type(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "svc_csv":
        model_type = "svc"
    elif callback_query.data == "logic_csv":
        model_type = "logistic"
    else:
        model_type = "undefined"
    await state.update_data(model_type=model_type)
    await callback_query.message.answer("Пришлите CSV файл.")
    await state.set_state(TrainCSVStates.waiting_for_csv)
    await callback_query.answer()

@router.message(TrainCSVStates.waiting_for_csv, F.document)
async def process_csv_file(message: types.Message, state: FSMContext):
    document = message.document

    if not document.file_name.endswith(".csv"):
        await message.answer("Пожалуйста, отправьте файл в формате CSV.")
        return

    user_data = await state.get_data()
    model_id = user_data.get('model_id')
    ml_model_type = user_data.get('model_type')

    if not model_id or not ml_model_type:
        await message.answer("Не были получены необходимые параметры модели. Пожалуйста, повторите процесс.")
        return

    file = await message.bot.get_file(document.file_id)
    file_url = f"https://api.telegram.org/file/bot{message.bot.token}/{file.file_path}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    file_content = await response.read()
                else:
                    await message.answer(f"Не удалось скачать файл. Статус: {response.status}.")
                    return
    except Exception as e:
        await message.answer(f"Ошибка при скачивании файла: {e}")
        return

    url = f"{API_URL}/fit_csv?model_id={model_id}&ml_model_type={ml_model_type}"
    data = aiohttp.FormData()
    data.add_field('file', BytesIO(file_content), filename=document.file_name, content_type='text/csv')

    async with aiohttp.ClientSession() as session:
        try:
            response = await session.post(url, data=data)
            if response.status == 200:
                result = await response.json()

                # Проверка типа данных в ответе, чтобы убедиться, что это не список
                if isinstance(result, list):
                    # Если это список, обработаем как список
                    result_message = "\n".join([item.get("message", "Без сообщения") for item in result])
                    await message.answer(f"Модель обучена: {result_message}")
                else:
                    # Если это словарь, обработаем как словарь
                    await message.answer(f"Ошибка при обучении модели: {result.get('message', 'Нет сообщения')}")
            else:
                await message.answer(f"Ошибка: {response.text}")
        except Exception as e:
            await message.answer(f"Ошибка при вызове API: {str(e)}")

@router.message(Command("predict"))
async def cmd_predict(message: types.Message, state: FSMContext):
    await message.answer("Введите идентификатор модели для предсказания:")
    await state.set_state(PredictStates.waiting_for_model_id)

@router.message(PredictStates.waiting_for_model_id)
async def process_predict_model_id(message: types.Message, state: FSMContext):
    await state.update_data(model_id=message.text)
    await message.answer(
        "Введите признаки для предсказания.\n"
        "Формат: строки, разделённые точкой с запятой, значения через запятую.\n"
        "Пример: 1,2,3; 4,5,6"
    )
    await state.set_state(PredictStates.waiting_for_features)

@router.message(PredictStates.waiting_for_features)
async def process_predict_features(message: types.Message, state: FSMContext):
    try:
        rows = message.text.split(";")
        features = [list(map(float, row.split(", "))) for row in rows if row.strip()]
    except Exception:
        await message.answer("Ошибка при обработке признаков. Проверьте формат ввода.")
        return
    data = await state.get_data()
    payload = {
        "id": data["model_id"],
        "X": features
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                if predictions:
                    text = "\n".join([f"Метка: {p['label']}, вероятность: {p['probability']}" for p in predictions])
                else:
                    text = "Нет предсказаний."
                await message.answer(f"Предсказания:\n{text}")
            else:
                await message.answer(f"Ошибка: {response.text}")
        except Exception as e:
            await message.answer(f"Ошибка при вызове API: {str(e)}")
    await state.clear()

@router.message(Command("list"))
async def cmd_list(message: types.Message):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/models")
            if response.status_code == 200:
                result = response.json()
                models = result[0].get("models", [])
                if models:
                    text = "\n".join([f"ID: {m['id']}, тип: {m['type']}" for m in models])
                else:
                    text = "Нет доступных моделей."
                await message.answer(text)
            else:
                await message.answer(f"Ошибка: {response.text}")
        except Exception as e:
            await message.answer(f"Ошибка при вызове API: {str(e)}")

@router.message(Command("status"))
async def cmd_status(message: types.Message):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/get_status")
            if response.status_code == 200:
                result = response.json()
                statuses = "\n".join([s["status"] for s in result])
                await message.answer(f"Статус моделей:\n{statuses}")
            else:
                await message.answer(f"Ошибка: {response.text}")
        except Exception as e:
            await message.answer(f"Ошибка при вызове API: {str(e)}")

@router.message(Command("load"))
async def cmd_load(message: types.Message, state: FSMContext):
    await message.answer("Введите идентификатор модели для загрузки:")
    await state.update_data(action="load")
    await state.set_state(ModelIDState.waiting_for_model_id)

@router.message(Command("unload"))
async def cmd_unload(message: types.Message, state: FSMContext):
    await message.answer("Введите идентификатор модели для выгрузки:")
    await state.update_data(action="unload")
    await state.set_state(ModelIDState.waiting_for_model_id)

@router.message(Command("remove"))
async def cmd_remove(message: types.Message, state: FSMContext):
    await message.answer("Введите идентификатор модели для удаления:")
    await state.update_data(action="remove")
    await state.set_state(ModelIDState.waiting_for_model_id)

@router.message(ModelIDState.waiting_for_model_id)
async def process_model_id_action(message: types.Message, state: FSMContext):
    data = await state.get_data()
    action = data.get("action")
    model_id = message.text
    async with httpx.AsyncClient() as client:
        try:
            if action == "load":
                payload = {"id": model_id}
                response = await client.post(f"{API_URL}/load", json=payload)
            elif action == "unload":
                payload = {"id": model_id}
                response = await client.post(f"{API_URL}/unload", json=payload)
            elif action == "remove":
                response = await client.delete(f"{API_URL}/remove/{model_id}")
            else:
                await message.answer("Неизвестное действие.")
                await state.clear()
                return

            if response.status_code == 200:
                result = response.json()
                await message.answer(f"Успех: {result[0]['message']}")
            else:
                await message.answer(f"Ошибка: {response.text}")
        except Exception as e:
            await message.answer(f"Ошибка при вызове API: {str(e)}")
    await state.clear()

@router.message(Command("remove_all"))
async def cmd_remove_all(message: types.Message, state: FSMContext):
    await message.answer("Вы уверены, что хотите удалить все модели? (Да/Нет)")
    await state.set_state(RemoveAllStates.waiting_for_confirmation)

@router.message(RemoveAllStates.waiting_for_confirmation)
async def process_remove_all_confirmation(message: types.Message, state: FSMContext):
    if message.text.lower() == "да":
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(f"{API_URL}/remove_all")
                if response.status_code == 200:
                    result = response.json()
                    texts = "\n".join([r["message"] for r in result])
                    await message.answer(f"Все модели удалены:\n{texts}")
                else:
                    await message.answer(f"Ошибка: {response.text}")
            except Exception as e:
                await message.answer(f"Ошибка при вызове API: {str(e)}")
    else:
        await message.answer("Удаление отменено.")
    await state.clear()

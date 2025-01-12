import streamlit as st
import aiohttp
import pandas as pd 
import json
import asyncio
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

url_get_root = 'http://0.0.0.0:8000/'
url_get_status = 'http://0.0.0.0:8000/api/v1/models/get_status'
url_fit = 'http://0.0.0.0:8000/api/v1/models/fit'
url_load = 'http://0.0.0.0:8000/api/v1/models/load'
url_predict = 'http://0.0.0.0:8000/api/v1/models/predict'
url_unload = 'http://0.0.0.0:8000/api/v1/models/unload'
url_list_models = 'http://0.0.0.0:8000/api/v1/models/models'
url_remove_id = 'http://0.0.0.0:8000/api/v1/models/remove/'
url_remove_all = 'http://0.0.0.0:8000/api/v1/models/remove_all'

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return response, await response.json()
        
async def post_data(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            return response, await response.json()
        
async def post_data_predict(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            try:
                # Пробуем распарсить ответ как JSON
                return response, await response.json()
            except aiohttp.ContentTypeError:
                # Если не удается распарсить как JSON, возвращаем текст
                return response, await response.text()

async def delete_id_model(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.delete(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            return response, await response.json()
        
async def delete_models(url):
    async with aiohttp.ClientSession() as session:
        async with session.delete(url) as response:
            return response, await response.json()

async def main():
    st.title("Анализ медицинских данных для мониторинга здоровья")
    st.subheader("Цель проекта: Разработка системы классификации ЭКГ для определения состояния здоровья человека. В данном проекте предлагается использовать открытые данные ЭКГ для анализа здоровья человека.")

     

    #EDA
    

    st.header("Доступные функции:")

    st.subheader("Получение статуса:")
    if st.button("Получить статус"):
        response, result_response = await fetch_data(url_get_status)
        if response.status == 200:
            st.markdown("Статус: " + str(result_response))
        else:
            st.markdown("Не удалось получить статус:")
            st.markdown(str(result_response))

    st.subheader("Обучение модели")

    id = st.text_input("Введите id модели", None, key="fit_id")
    ml_model_type = st.selectbox("Выберите тип модели для обучения", ("logistic", "svc"), None, key="fit_type")
    hyperparameters = st.text_area("Введите гиперпараметры модели", '{"C": 0.1}', key="fit_hyperparameters")

    st.markdown("Выберите json файл-json с данными для обучения")
    uploaded_file = st.file_uploader("Выберите json-файл", type=["json"], key="data_train")
    data=None
    if uploaded_file is not None:
        data = uploaded_file.read()
        st.write("Файл загружен")
        data_in_file = json.loads(data)
    else:
        st.write("**Выберите файл!**")   

    if st.button("Обучить"):
        if data is not None and id is not None and ml_model_type is not None:
            hyper = json.loads(hyperparameters)
            data_in_file["config"] = {"id": id, "ml_model_type": ml_model_type, "hyperparameters": hyper}

            json_data = json.dumps(data_in_file)
   
            st.write("Обучение модели в процессе.")
            response, result_response = await post_data(url_fit, json_data)
            
            if response.status == 200:
                st.markdown(f"Модель {id} обучена и сохранена")
            else:
                st.markdown("Не получилось обучить модель.")
                st.markdown(str(result_response))
        else:
            if data is None:
                st.markdown("Выберите файл для обучения")
            if id is None:
                st.markdown("Напишите id для обучения модели")
            if ml_model_type is None:
                st.markdown("Выберите тип модели")

    st.subheader("Список моделей:")
    if st.button("Получить"):
        response, result_response = await fetch_data(url_list_models)
        if response.status == 200:
            
            for m in result_response:
                if len(m['models']) > 0:
                    st.markdown("Список моделей")
                    for models in m['models']:
                        model = "логистической регрессии"
                        if models['type'] == 'svc':
                            model = "SVC"
                        st.markdown(f"  Модель {model} с идентификатором '{models['id']}';")
                else:
                    st.markdown("Моделей нет.")
        else:
            st.markdown("Не удалось получить список моделей:")
            st.markdown(str(result_response))

    st.subheader("Загрузка модели")
    id = st.text_input("Введите id ранее обученной модели", None, key="load_id")
    if st.button("Загрузить модель"):
        if id is not None:
            data = json.dumps({"id": id})
            response, result_response = await post_data(url_load, data)
            if response.status == 200:
                st.markdown(f"Модель '{id}' была загружена")
            elif response.status == 422:
                st.markdown(f"Модель '{id}' не была загружена")
                if 'not found file of model' in result_response['detail'][0]['msg']:
                    st.markdown(f"Модель '{id}' не была найдена. Возможно она не была обучена.")
                else:
                    st.markdown(str(result_response))
            else:
                    st.markdown(str(result_response))

    #500 Internal Server Error
    st.subheader("Анализ данных с помощью модели")
    id_pred = st.text_input("Введите id ранее загруженной модели", None, key="pred_id")
    uploaded_file_pred = st.file_uploader("Выберите json-файл с данными", type=["json"], key="data_predict")
    if uploaded_file_pred is not None:
        data = uploaded_file_pred.read()
        st.write("Файл загружен")
        data_in_file = json.loads(data)
    else:
        st.write("**Выберите файл!**")  

    if st.button("Анализ"):
        if uploaded_file_pred is not None and id_pred is not None:
            data_in_file["id"] = id_pred
            json_data = json.dumps(data_in_file)
            response, result_response = await post_data_predict(url_predict, json_data)
            if response.status == 200:
                st.markdown("Анализ ЭКГ:")
                for prediction in result_response['predictions']:
                    label = "В данных не найдено отклонений"
                    if prediction['label'] == "PROBLEM":
                        label = "В данных **возможны** отклонения"
                    st.markdown(f"{label} с вероятностью {round(prediction['probability'], 4)}")
            elif response.status == 422:
                st.markdown("Не удалось провести анализ данных.")
                st.markdown("Данные  имеют некорректный формат.")
            else:
                st.markdown("Не удалось провести анализ данных.")
                if 'not found' in result_response['detail'][0]['msg']:
                    st.markdown(f"Модель {id_pred} не найдена. Для её использования, пожалуйста, сначала загрузите её.")
                else:
                    st.markdown(str(result_response))

    st.subheader("Построение кривых обучения")
    id_roc = st.text_input("Введите id ранее обученной модели", None, key="roc_id")
    st.markdown("Выберите json файл-json с данными для построения кривых")
    uploaded_file = st.file_uploader("Выберите json-файл", type=["json"], key="roc_train")
    data=None
    if uploaded_file is not None:
        data_file = uploaded_file.read()
        st.write("Файл загружен")
        data_in_file = json.loads(data_file)
    else:
        st.write("**Выберите файл!**")
    if st.button("Построение"):
        if uploaded_file is not None and id_roc is not None:
            data_for_request = {'id':id_roc, 'X':data_in_file['X']}
            json_data = json.dumps(data_for_request)
            response, result_response = await post_data_predict(url_predict, json_data)
            if response.status == 200:
                
                y_test = data_in_file["y"]
                y_probs = []
                y_pred = []
                for prediction in result_response['predictions']:
                    y_probs.append(prediction['probability'])
                    y_pred.append(prediction['label'])

                roc_auc = roc_auc_score(y_test, y_probs)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="binary", pos_label="PROBLEM")
                recall = recall_score(y_test, y_pred, average="binary", pos_label="PROBLEM")
                f1 = f1_score(y_test, y_pred, average="binary", pos_label="PROBLEM")
                
                st.markdown(f"AUC - ROC Score: {roc_auc:.2f}")
                st.markdown(f"Accuracy: {accuracy:.2f}")
                st.markdown(f"Precision: {precision:.2f}")
                st.markdown(f"Recall: {recall:.2f}")
                st.markdown(f"F1 Score: {f1:.2f}")
                
                fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label="PROBLEM")
                roc_auc = auc(fpr, tpr)

                fig = plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Кривые обучения)')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Ложно положительный рейтинг')
                plt.ylabel('Верно положительный рейтинг')
                plt.title('ROC-кривые')
                plt.legend(loc="lower right")
                plt.show()
                st.pyplot(fig)
                
            elif response.status == 422:
                st.markdown("Не удалось провести анализ данных.")
                st.markdown("Данные  имеют некорректный формат.")
            else:
                st.markdown("Не удалось провести анализ данных.")
                if 'not found' in result_response['detail'][0]['msg']:
                    st.markdown(f"Модель {id_pred} не найдена. Для её использования, пожалуйста, сначала загрузите её.")
                else:
                    st.markdown(str(result_response))


        else:
            if data_file is None:
                st.write("Выберите файл для построения кривых.")
            if id_roc is None:
                st.write("Выберите модель.")

    st.subheader("Выгрузка модели")
    id = st.text_input("Введите id ранее загруженной модели", None, key="unload_id")
    if st.button("Выгрузить модель"):
        if id is not None:
            data = json.dumps({"id": id})
            response, result_response = await post_data(url_unload, data)
            if response.status == 200:
                st.markdown(f"Модель '{id}' была выгружена")
            elif response.status == 422:
                st.markdown(f"Модель '{id}' не была выгружена")
                st.markdown(f"Модель '{id}' не была найдена. Возможно она не была загружена.")
            else:
                    st.markdown(str(result_response))
    
    st.subheader("Удаление модели")
    id = st.text_input("Введите id модели для удаления", None, key="delete_id")
    if st.button("Удалить модель"):
        if id is not None:
            data = json.dumps({"id": id})
            response, result_response = await delete_id_model(url_remove_id+str(id), data)
            if response.status == 200:
                st.markdown(f"Модель '{id}' была удалена.")
            elif response.status == 422:
                st.markdown(f"Модель '{id}' не была удалена.")
                if 'not found model' in result_response['detail'][0]['msg']:
                    st.markdown(f"Модель '{id}' не была найдена. Возможно она не была обучена.")
                else:
                    st.markdown(str(result_response))
            else:
                st.markdown(f"Модель '{id}' не была удалена.")
                if 'not found model' in result_response['detail'][0]['msg']:
                    st.markdown(f"Модель '{id}' не была найдена. Возможно она не была обучена.")
                else:
                    st.markdown(str(result_response))

    st.subheader("Удалить все модели")
    if st.button("Удалить все модели"):
        response, result_response = await delete_models(url_remove_all)
        if response.status == 200:
                st.markdown(f"Все модели были удалены.")
        else:
                st.markdown(f"Модели не были удалены.")
                st.markdown(str(result_response))
        


if __name__=="__main__":
    asyncio.run(main())






        






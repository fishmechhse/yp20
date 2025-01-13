import streamlit as st
import json
from api_func import post_data, post_data_predict
from api_path import url_load, url_predict, url_unload, url_fit
st.title("Анализ данных:")
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
        response, result_response = post_data(url_fit, json_data)
        
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

st.subheader("Загрузка модели")
id = st.text_input("Введите id ранее обученной модели", None, key="load_id")
if st.button("Загрузить модель"):
    if id is not None:
        data = json.dumps({"id": id})
        response, result_response = post_data(url_load, data)
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
        response, result_response = post_data_predict(url_predict, json_data)
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

st.subheader("Выгрузка модели")
id = st.text_input("Введите id ранее загруженной модели", None, key="unload_id")
if st.button("Выгрузить модель"):
    if id is not None:
        data = json.dumps({"id": id})
        response, result_response = post_data(url_unload, data)
        if response.status == 200:
            st.markdown(f"Модель '{id}' была выгружена")
        elif response.status == 422:
            st.markdown(f"Модель '{id}' не была выгружена")
            st.markdown(f"Модель '{id}' не была найдена. Возможно она не была загружена.")
        else:
                st.markdown(str(result_response))
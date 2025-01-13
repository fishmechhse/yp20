from api_func import fetch_data, delete_id_model, delete_models
import streamlit as st
import json
from api_path import url_get_status, url_list_models, url_remove_id, url_remove_all
st.title("Описание моделей:")

st.subheader("Список моделей:")
if st.button("Получить"):
    response, result_response = fetch_data(url_list_models)
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
st.subheader("Удаление модели")
id = st.text_input("Введите id модели для удаления", None, key="delete_id")
if st.button("Удалить модель"):
    if id is not None:
        data = json.dumps({"id": id})
        response, result_response = delete_id_model(url_remove_id+str(id), data)
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
    response, result_response =delete_models(url_remove_all)
    if response.status == 200:
            st.markdown(f"Все модели были удалены.")
    else:
            st.markdown(f"Модели не были удалены.")
            st.markdown(str(result_response))




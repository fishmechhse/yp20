import streamlit as st
import aiohttp
import pandas as pd 
import json
import requests
import asyncio
import seaborn as sns
import matplotlib.pyplot as plt

url_get_root = 'http://0.0.0.1:8000/'
url_get_status = 'http://0.0.0.1:8000/api/v1/models/get_status'
url_fit = 'http://0.0.0.1:8000/api/v1/models/fit'
url_load = 'http://0.0.0.1:8000/api/v1/models/load'
url_predict = 'http://0.0.0.1:8000/api/v1/models/predict'
url_unload = 'http://0.0.0.1:8000/api/v1/models/unload'
url_list_models = 'http://0.0.0.1:8000/api/v1/models/models'
url_remove_id = 'http://0.0.0.1:8000/api/v1/models/remove/'
url_remove_all = 'http://0.0.0.1:8000/api/v1/models/remove_all'

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
        
async def post_data(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            return await response.json()
        
async def post_data_predict(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            try:
                # Пробуем распарсить ответ как JSON
                return await response.json()
            except aiohttp.ContentTypeError:
                # Если не удается распарсить как JSON, возвращаем текст
                return await response.text()

async def delete_id_model(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.delete(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            return await response.json()
        
async def delete_models(url):
    async with aiohttp.ClientSession() as session:
        async with session.delete(url) as response:
            return await response.json()

async def main():
    st.title("Анализ медицинских данных для мониторинга здоровья")
    st.subheader("Цель проекта: Разработка системы классификации ЭКГ для определения состояния здоровья человека. В данном проекте предлагается использовать открытые данные ЭКГ для анализа здоровья человека.")

    st.header("Выберите json файл-json с данными для обучения и csv-файл с данными для EDA")

    uploaded_file = st.file_uploader("Выберите json-файл", type=["json"], key="data_train")
    uploaded_csv_file = st.file_uploader("Выберите csv-файл", type="csv")
    if uploaded_file is not None and uploaded_csv_file is not None:
        data = uploaded_file
        df = uploaded_csv_file
        st.write("Файлы загружен")

        df = pd.read_csv(uploaded_csv_file)
        st.write('Первые 5 строк')
        st.write(df.head())
        st.write('Описательная статистика')
        st.write(df.describe(include='all'))
        x = df.drop('target', axis=1)
        st.subheader("Тепловая карта корреляции")
        plt.figure(figsize=(60, 50), dpi=300)
        sns.heatmap(x.corr(numeric_only=True),
                    linewidths=0.5, annot=True,
                    cmap='Blues', linecolor="white",
                    annot_kws={"size": 12}, cbar_kws={"shrink": 1})
        st.pyplot(plt)
        plt.close()
        st.subheader('Распределение диагнозов')
        y = df['target'].value_counts()
        num = y / y.sum() * 100
        plt.figure(figsize=(8, 6))
        num.plot(kind='bar', color=['green', 'orange'])
        plt.xlabel('Диагноз')
        plt.ylabel('Процент')
        plt.xticks(rotation=0)
        plt.ylim(0, 100)
        for index, value in enumerate(num):
            plt.text(index, value + 1, f'{value:.1f}%', ha='center')
        st.pyplot(plt)
        plt.close()
        st.subheader('Гистограма возрастного распределения')
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=10,
                     palette={'NORM': 'green', 'PROBLEM': 'orange'})
        plt.xlabel('Возраст')
        plt.ylabel('Количество')
        plt.legend(['Problem', 'Norm'], loc='upper right', title='Категории', frameon=True)
        st.pyplot(plt)
        plt.close()
        target = df.columns[-1]
        features = df.columns[:-1]
        num_f = len(features)
        cols = 4
        rows = (num_f // cols) + (num_f % cols > 0)
        st.subheader("Гистограммы распределения признаков по целевой переменной")
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
        axes = axes.flatten()
        for i, k in enumerate(features):
            sns.histplot(data=df, x=k, hue=target, multiple="stack", kde=True, bins=30, ax=axes[i])
            axes[i].set_title(k)
            axes[i].set_xlabel(k)
            axes[i].set_ylabel('Количество')
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        num_f = len(features)
        cols = 4
        rows = (num_f // cols) + (num_f % cols > 0)
        st.subheader("Boxplots распределения признаков по целевой переменной")
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
        axes = axes.flatten()
        for i, k in enumerate(features):
            sns.boxplot(data=df, x=target, y=k, ax=axes[i])
            axes[i].set_title(k)
            axes[i].set_xlabel(target)
            axes[i].set_ylabel(k)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)
        

        st.header("Доступные функции:")

        st.subheader("Путь:")
        if st.button("Получить root"):
            response = await fetch_data(url_get_root)
            st.markdown(response)

        st.subheader("Получение статуса:")
        if st.button("Получить статус"):
            response = await fetch_data(url_get_status)
            st.markdown(response)

        st.subheader("Обучение модели")
        if st.button("Обучить"):
            st.write("Обучение модели в процессе.")
            response = await post_data(url_fit, data)
            st.markdown(response[0]['message'])

        st.subheader("Список моделей:")
        if st.button("Получить"):
            response = await fetch_data(url_list_models)
            st.markdown(response[0]['models'])


        st.subheader("Загрузка модели")
        id = st.text_input("Введите id ранее обученной модели", None, key="load_id")
        if st.button("Загрузить модель"):
            if id is not None:
                data = json.dumps({"id": id})
                response = await post_data(url_load, data)
                st.markdown(str(response[0]['message']))

        #500 Internal Server Error
        st.subheader("Анализ данных с помощью модели")
        uploaded_file_pred = st.file_uploader("Выберите json-файл", type=["json"], key="data_predict")
        if st.button("Анализ"):
            if uploaded_file is not None:
                data_req = uploaded_file_pred #json.dumps(json.load(uploaded_file_pred))
                st.write("Файл загружен")
                response = await post_data_predict(url_predict, data_req)
                st.markdown(str(response))


        st.subheader("Выгрузка модели")
        id = st.text_input("Введите id ранее загруженной модели", None, key="unload_id")
        if st.button("Выгрузить модель"):
            if id is not None:
                data = json.dumps({"id": id})
                response = await post_data(url_unload, data)
                st.markdown(str(response[0]['message']))
            #response = await post_data(url_fit, data)
            #st.markdown(str(response))
        
        st.subheader("Удаление модели")
        id = st.text_input("Введите id модели для удаления", None, key="delete_id")
        if st.button("Удалить модель"):
            if id is not None:
                data = json.dumps({"id": id})
                response = await delete_id_model(url_remove_id+str(id), data)
                st.markdown(str(response[0]['message']))

        st.subheader("Удалить все модели")
        if st.button("Удалить все модели"):
            response = await delete_models(url_remove_all)
            st.markdown(str(response[0]['message']))
            
    else:
        st.write("**Выберите файл!**")

if __name__=="__main__":
    asyncio.run(main())






        






import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

st.title("EDA")
st.markdown("Для получения EDA выберите csv-файл.")
uploaded_csv_file = st.file_uploader("Выберите csv-файл", type="csv")

if uploaded_csv_file is not None:
    df = uploaded_csv_file
    st.markdown("Файл загружен")

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
else:
    st.write("**Выберите файл!**")
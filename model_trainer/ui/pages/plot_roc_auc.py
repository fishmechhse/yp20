
import streamlit as st
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from api_func import post_data_predict, post_data
from api_path import url_load, url_predict, url_unload
import pandas as pd

st.title("Построение кривых обучения:")
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
            if 'not found' in dict(result_response['detail'])[0]['msg']:
                st.markdown(f"Модель '{id}' не была найдена. Возможно она не была обучена.")
            else:
                st.markdown(str(result_response))
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
        response, result_response = post_data_predict(url_predict, json_data)
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

            cm = confusion_matrix(y_test, y_pred, labels=["NORM", "PROBLEM"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["NORM", "PROBLEM"])
            disp.plot()
            plt.show()
            st.pyplot(plt)
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, target_names=["NORM", "PROBLEM"], output_dict=True)).transpose())

            fpr_p, tpr_p, thresholds = roc_curve(y_test, y_probs, pos_label="PROBLEM")
            fpr_n, tpr_n, thresholds = roc_curve(y_test, y_probs, pos_label="NORM")
            roc_auc = auc(fpr_p, tpr_p)

            fig = plt.figure()
            plt.title(f'Receiver Operating Characteristic')
            plt.plot(fpr_p, tpr_p, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            st.pyplot(fig)
            
        elif response.status == 422:
            st.markdown("Не удалось провести анализ данных.")
            st.markdown("Данные  имеют некорректный формат.")
        else:
            st.markdown("Не удалось провести анализ данных.")
            try:
                if 'not found' in result_response['detail'][0]['msg']:
                    st.markdown(f"Модель {id_roc} не найдена. Для её использования, пожалуйста, сначала загрузите её.")
            except:
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
        response, result_response = post_data(url_unload, data)
        if response.status == 200:
            st.markdown(f"Модель '{id}' была выгружена")
        elif response.status == 422:
            st.markdown(f"Модель '{id}' не была выгружена")
            st.markdown(f"Модель '{id}' не была найдена. Возможно она не была загружена.")
        else:
                st.markdown(str(result_response))
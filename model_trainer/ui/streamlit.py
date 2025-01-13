import streamlit as st
    
pg = st.navigation([st.Page("pages/main_page.py", title="Описание проекта"), st.Page("pages/eda.py", title="EDA"), st.Page("pages/info_of_models.py", title="Описание моделей"), st.Page("pages/use_models.py", title="Анализ данных"),st.Page("pages/plot_roc_auc.py", title="Построение кривых обучения")])
pg.run()



        






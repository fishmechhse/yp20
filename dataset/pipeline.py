import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder

from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import joblib

from trained_pipeline.trim_transformer import make_data, TrimTransformer


def run_pipeline():
    sX_train, sX_test, sy_train, sy_test = make_data()

    num_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

    column_values_caster_transformer = Pipeline(steps=[('trimer', TrimTransformer())])
    target_transformer = Pipeline(steps=[('target_enc', TargetEncoder(cols=['name'], smoothing=1))])
    # Преобразование числовых столбцов
    numerical_transformer = Pipeline(steps=[
        ('trimer', column_values_caster_transformer),
        ('imputer', SimpleImputer(strategy='mean')),  # Замена пропусков на среднее
        ('scaler', StandardScaler())  # Масштабирование признаков
    ])

    # Преобразование категориальных столбцов
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),  # Замена пропусков на 'NA'
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),  # OHE-кодирование
    ])

    # hasherTransformer = HasherTransformer(column="name", n_features=100)
    # Объединяем преобразования с помощью ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", target_transformer, ['name']),
            ('num', numerical_transformer, num_columns),
            ('cat', categorical_transformer, cat_columns)
        ]
    )

    parameters_grid = {"alpha": np.arange(1, 10.1 + 1e-5, 1)}

    e_net = Ridge()
    grid = GridSearchCV(estimator=e_net,
                        param_grid=parameters_grid,
                        scoring='r2',
                        cv=10,
                        n_jobs=-1,
                        verbose=3)

    # Полный пайплайн с линейной регрессией
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', grid)
    ])

    # 3. Обучение модели
    pipeline.fit(sX_train, sy_train)

    # 4. Оценка модели
    sy_pred = pipeline.predict(sX_test)
    sy_train_pred = pipeline.predict(sX_train)

    grid_search = pipeline.named_steps["regressor"]

    features = pipeline[:-1].get_feature_names_out()
    d = {'col1': grid_search.best_estimator_.coef_, 'col2': features}
    tdf = pd.DataFrame(data=d)
    print(tdf.sort_values(by=["col2"]))

    best_estimator = grid_search.best_estimator_

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"test MSE: {MSE(sy_test, sy_pred)}")
    print(f"test r2 score = {r2_score(sy_test, sy_pred)}")
    print(f"[train] MSE: {MSE(sy_train, sy_train_pred)}")
    print(f"[train] r2 score = {r2_score(sy_train, sy_train_pred)}")
    print(f"coef_: {best_estimator.coef_}")
    return pipeline


def save_estimator(pipeline, filename):
    with open(filename, 'wb') as f:
        joblib.dump(pipeline, f)


def get_pipeline(pkl_file):
    clf = joblib_load_estimator(pkl_file)
    return clf


def joblib_load_estimator(filename):
    f = open(filename, "rb")
    model = joblib.load(f)
    return model


if __name__ == '__main__':
    pipeline = run_pipeline()
    save_estimator(pipeline, './../model/pipeline.pkl')

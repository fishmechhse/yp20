import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.multioutput import MultiOutputRegressor


def create_target(scp_superclass):
    if ('MI' in scp_superclass):
        return 0
    if ('NORM' in scp_superclass):
        return 1
    else:
        return 2


if __name__ == '__main__':
    df = pd.read_csv('v5_v2_ecg_extracted_features.csv',
                     converters={'scp_superclass': pd.eval, 'scp_superclass_len': pd.eval })
    columns = []
    interval_columns = []
    for column in df.columns:
        if ('_stat_' in column) | ('_interval_' in column):
            columns.append(column)

    columns.append('ECG_Rate_mean')
    columns.append('heart_axis')
    columns.append('scp_superclass')
    columns.append('validated_by_human')
    columns.append('baseline_drift')
    columns.append('age')
    columns.append('patient_id')
    columns.append('ecg_id')
    columns.append('burst_noise')
    tmp_df = df[columns]
    datatypes = tmp_df.dtypes
    tmp_df = tmp_df[tmp_df['burst_noise'].isnull()].drop(columns=['burst_noise'])
    tmp_df = tmp_df[tmp_df['baseline_drift'].isnull()].drop(columns=['baseline_drift'])
    tmp_df = tmp_df[tmp_df['validated_by_human'] == True].drop(columns=['validated_by_human'])

    # Step 3: Use apply to create a new column based on 'Age'
    tmp_df['target'] = tmp_df['scp_superclass'].apply(create_target)

    tmp_df.drop(columns=interval_columns, inplace=True)
    tmp_df.drop(columns=['scp_superclass', 'ecg_id', 'patient_id'], inplace=True)
    tmp_df = tmp_df.dropna()
    print(tmp_df['target'].value_counts())
    y = tmp_df['target']
    tmp_df = tmp_df.round(4)
    X = tmp_df.drop(columns=['target'])

    # mlb = MultiLabelBinarizer()
    # y_binarized = mlb.fit_transform(y)
    # Step 3: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), ['heart_axis']),  # Apply OneHotEncoder to heart_axis column
        ],
        remainder='passthrough'  # Keep other columns unchanged if any
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Step to scale features
        ('scaler', StandardScaler()),  # Step to scale features
        ('logistic',
         CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False))])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('RandomForestClassifier')
    print(classification_report(y_test, y_pred))

    # Step 5: Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Step 6: Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

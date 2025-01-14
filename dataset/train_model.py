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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from dataset.pca import plot_projection, make_pca


def create_target(scp_superclass):
    if ('NORM' in scp_superclass) and (len(scp_superclass) == 1):
        return 'NORM'
    else:
        return 'PROBLEM'


def print_roc_auc(y_test, y_probability, label):
    print(roc_auc_score(y_test, y_probability))

    y_test = y_test.map({'NORM': 0, 'PROBLEM': 1}).astype(int)
    fpr, tpr, threshold = roc_curve(y_test, y_probability)
    roc_auc = auc(fpr, tpr)

    plt.title(f'{label}Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('v5_v2_ecg_extracted_features.csv', converters={'scp_superclass': pd.eval})
    columns = []
    for column in df.columns:
        if ('stat_' in column) | ('_interval_' in column):
            columns.append(column)

    columns.append('ECG_Rate_mean')
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
    tmp_df.drop(columns=['scp_superclass', 'ecg_id', 'patient_id'], inplace=True)
    tmp_df = tmp_df.dropna()
    print(tmp_df['target'].value_counts())
    y = tmp_df['target']
    tmp_df = tmp_df.round(4)
    X = tmp_df.drop(columns=['target'])

    X_np = X.to_numpy().tolist()
    y_np = y.to_numpy().tolist()

    train_set = {
        "X": X_np,
        "y": y_np,
        "config": {
            "id": "string",
            "ml_model_type": "svc",
            "hyperparameters": {
                'C': 0.1
            }
        }
    }
    with open('train_for_request.json', 'w') as json_file:
        json.dump(train_set, json_file, indent=4)

    test_set = {
        "id": "111",
        "X": X_np[:10]
    }
    with open('test_request.json', 'w') as json_file:
        json.dump(test_set, json_file, indent=4)

    tmp_df.to_csv('clean_data_for_training.csv', index=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    make_pca(X, y)

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Step to scale features
        ('logistic', LogisticRegression(solver='liblinear', max_iter=1200, class_weight='balanced'))
        # Step to fit logistic regression model
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('regression')
    print(classification_report(y_test, y_pred))

    y_probability = pipeline.predict_proba(X_test)[:, 1]
    print_roc_auc(y_test, y_probability, 'Logistic Regression: ')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Step to scale features
        ('logistic', SVC(class_weight='balanced', probability=True))  # Step to fit logistic regression model
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('svc')
    print(classification_report(y_test, y_pred))

    y_probability = pipeline.predict_proba(X_test)[:, 1]
    print_roc_auc(y_test, y_probability, 'SVC model: ')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

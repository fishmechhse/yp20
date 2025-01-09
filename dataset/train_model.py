import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def create_target(scp_superclass):
    if 'NORM' in scp_superclass:
        return 0
    else:
        return 1


if __name__ == '__main__':
    df = pd.read_csv('v5_v2_ecg_extracted_features.csv')
    columns = []
    for column in df.columns:
        if ('_stat_' in column) | ('_interval_' in column):
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
    tmp_df = tmp_df[tmp_df['burst_noise'].isnull()].drop(columns=['burst_noise'])
    tmp_df = tmp_df[tmp_df['baseline_drift'].isnull()].drop(columns=['baseline_drift'])
    tmp_df = tmp_df[tmp_df['validated_by_human'] == True].drop(columns=['validated_by_human'])





    # Step 3: Use apply to create a new column based on 'Age'
    tmp_df['target'] = tmp_df['scp_superclass'].apply(create_target)
    tmp_df.drop(columns=['scp_superclass', 'ecg_id', 'patient_id'], inplace=True)
    tmp_df = tmp_df.dropna()
    print(tmp_df['target'].value_counts())
    y = tmp_df['target']
    X = tmp_df.drop(columns=['target'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Step to scale features
        ('logistic', LogisticRegression(solver='liblinear', max_iter=600, class_weight='balanced'))  # Step to fit logistic regression model
    ])
    pipeline.fit(X_train, y_train)


    y_pred = pipeline.predict(X_test)
    print('regression')
    print(classification_report(y_test, y_pred))


    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Step to scale features
        ('logistic', SVC(class_weight='balanced'))  # Step to fit logistic regression model
    ])
    pipeline.fit(X_train, y_train)


    y_pred = pipeline.predict(X_test)
    print('svc')
    print(classification_report(y_test, y_pred))


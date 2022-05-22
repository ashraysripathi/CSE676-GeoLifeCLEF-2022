from tensorflow.keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import os
from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from GLC.metrics import predict_top_30_set
from GLC.submission import generate_submission_file
from GLC.metrics import top_k_error_rate_from_sets
from GLC.metrics import top_30_error_rate
import tensorflow.keras.optimizers
import matplotlib as plt

SUBMISSION_PATH = Path("submissions")
os.makedirs(SUBMISSION_PATH, exist_ok=True)
DATA_PATH = Path("D:\DL_GLC")
df_obs_fr = pd.read_csv(DATA_PATH / "observations" /
                        "observations_fr_train.csv", sep=";", index_col="observation_id")
df_obs_us = pd.read_csv(DATA_PATH / "observations" /
                        "observations_us_train.csv", sep=";", index_col="observation_id")
df_obs = pd.concat((df_obs_fr, df_obs_us))

obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
obs_id_val = df_obs.index[df_obs["subset"] == "val"].values

y_train = df_obs.loc[obs_id_train]["species_id"].values
y_val = df_obs.loc[obs_id_val]["species_id"].values

n_val = len(obs_id_val)
print("Validation set size: {} ({:.1%} of train observations)".format(
    n_val, n_val / len(df_obs)))

df_obs_fr_test = pd.read_csv(DATA_PATH / "observations" /
                             "observations_fr_test.csv", sep=";", index_col="observation_id")
df_obs_us_test = pd.read_csv(DATA_PATH / "observations" /
                             "observations_us_test.csv", sep=";", index_col="observation_id")

df_obs_test = pd.concat((df_obs_fr_test, df_obs_us_test))

obs_id_test = df_obs_test.index.values

print("Number of observations for testing: {}".format(len(df_obs_test)))

# print(df_obs_test.head())

df_env = pd.read_csv(DATA_PATH / "pre-extracted" /
                     "environmental_vectors.csv", sep=";", index_col="observation_id")

X_train = df_env.loc[obs_id_train].values

X_val = df_env.loc[obs_id_val].values
X_test = df_env.loc[obs_id_test].values

# print(y_train)
imp = SimpleImputer(
    missing_values=np.nan,
    strategy="constant",
    fill_value=np.finfo(np.float32).min,
)
imp.fit(X_train)

X_train = imp.transform(X_train)
X_val = imp.transform(X_val)
X_test = imp.transform(X_test)
n_features = X_train.shape[1]
print(X_train)
print("Rescaling")
#X_train = X_train / X_train.max(axis=0)
#X_val = X_val / X_val.max(axis=0)
#X_test = X_test / X_test.max(axis=0)


model = Sequential()
model.add(Dense(128, activation='relu',
                kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(17037, activation='softmax'))
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=1, batch_size=256, verbose=1)


def batch_predict(predict_func, X, batch_size=1024):
    res = predict_func(X[:1])
    n_samples, n_outputs, dtype = X.shape[0], res.shape[1], res.dtype

    preds = np.empty((n_samples, n_outputs), dtype=dtype)

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        preds[i:i+batch_size] = predict_func(X_batch)

    return preds


def predict_func(X):
    y_score = model.predict_proba(X)
    s_pred = predict_top_30_set(y_score)
    return s_pred


s_val = batch_predict(predict_func, X_val, batch_size=1024)
score_val = top_k_error_rate_from_sets(y_val, s_val)

print("Top-30 error rate: {:.1%}".format(score_val))

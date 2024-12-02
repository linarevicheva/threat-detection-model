
import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt
import tensorflow as tf

url= "data/"

df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
df_small_noise_url = url + df_small_noise_url_suffix
df_small_noise = pd.read_csv(
    df_small_noise_url, parse_dates=True, index_col="timestamp"
)

df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = url + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
)

print("Training Data Columns:", df_small_noise.columns)
print("Test Data Columns:", df_daily_jumpsup.columns)


if 'value' in df_small_noise.columns:
    df_small_noise = df_small_noise[['value']]
else:
    df_small_noise = df_small_noise.iloc[:, 0:1]

if 'value' in df_daily_jumpsup.columns:
    df_daily_jumpsup = df_daily_jumpsup[['value']]
else:
    df_daily_jumpsup = df_daily_jumpsup.iloc[:, 0:1]

print("Selected Training Data Columns:", df_small_noise.columns)
print("Selected Test Data Columns:", df_daily_jumpsup.columns)


print(df_small_noise.head())

print(df_daily_jumpsup.head())

fig, ax = plt.subplots()
df_small_noise.plot(legend=False, ax=ax)
plt.show()
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
plt.show()

training_mean = df_small_noise.mean().values[0]
training_std = df_small_noise.std().values[0] 

df_training_value = (df_small_noise - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

TIME_STEPS = 288


def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.UpSampling1D(size=2),
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.UpSampling1D(size=2),
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        layers.Conv1D(filters=1, kernel_size=3, activation="linear", padding="same"),
    ]
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

threshold = np.percentile(train_mae_loss, 90)

print("Reconstruction error threshold: ", threshold)

plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.show()

df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()

x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

anomaly_scores = np.zeros(len(df_test_value))
for i in range(len(test_mae_loss)):
    anomaly_scores[i:i + TIME_STEPS] += test_mae_loss[i]

aggregated_threshold = np.percentile(anomaly_scores, 99)
anomalous_data_indices = np.where(anomaly_scores > aggregated_threshold)[0]

df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()

model.save('anomaly_detection_model.h5')

np.save('training_mean.npy', training_mean)
np.save('training_std.npy', training_std)
np.save('threshold.npy', threshold)

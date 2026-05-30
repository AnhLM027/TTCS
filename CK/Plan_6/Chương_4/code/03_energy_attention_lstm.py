# File extracted from: 03_energy_attention_lstm.ipynb
# Code cells and text outputs

# %% [cell 1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob, os, warnings

import random

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})

DATA_DIR   = '../data/energy_consumption'
SAVE_DIR   = '../results/energy_attention'
os.makedirs(SAVE_DIR, exist_ok=True)

LOOK_BACK  = 48    # 48 bước thời gian
EPOCHS     = 40
BATCH_SIZE = 64

print(f'TF: {tf.__version__}')

# --- OUTPUT ---
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1780119829.115677 1984501 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# I0000 00:00:1780119829.150794 1984501 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# TF: 2.21.0
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1780119829.967592 1984501 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# --------------

# %% [cell 2]
csv_files=glob.glob(f'{DATA_DIR}/**/*.csv',recursive=True)+glob.glob(f'{DATA_DIR}/*.csv')
print('Found:',csv_files)

# lấy file lớn nhất
df=pd.read_csv(sorted(csv_files,key=os.path.getsize,reverse=True)[0])

# parse datetime
df['Datetime']=pd.to_datetime(df['Datetime'])
df=df.sort_values('Datetime').reset_index(drop=True)

# xử lý NaN
df=df.ffill().bfill()

print('Shape:',df.shape)
print('Columns:',list(df.columns))
print('\nMissing values:\n',df.isna().sum())

df.head()

# --- OUTPUT ---
# Found: ['../data/energy_consumption/NI_hourly.csv', '../data/energy_consumption/EKPC_hourly.csv', '../data/energy_consumption/pjm_hourly_est.csv', '../data/energy_consumption/FE_hourly.csv', '../data/energy_consumption/PJM_Load_hourly.csv', '../data/energy_consumption/PJME_hourly.csv', '../data/energy_consumption/AEP_hourly.csv', '../data/energy_consumption/DEOK_hourly.csv', '../data/energy_consumption/DOM_hourly.csv', '../data/energy_consumption/DAYTON_hourly.csv', '../data/energy_consumption/COMED_hourly.csv', '../data/energy_consumption/DUQ_hourly.csv', '../data/energy_consumption/PJMW_hourly.csv', '../data/energy_consumption/NI_hourly.csv', '../data/energy_consumption/EKPC_hourly.csv', '../data/energy_consumption/pjm_hourly_est.csv', '../data/energy_consumption/FE_hourly.csv', '../data/energy_consumption/PJM_Load_hourly.csv', '../data/energy_consumption/PJME_hourly.csv', '../data/energy_consumption/AEP_hourly.csv', '../data/energy_consumption/DEOK_hourly.csv', '../data/energy_consumption/DOM_hourly.csv', '../data/energy_consumption/DAYTON_hourly.csv', '../data/energy_consumption/COMED_hourly.csv', '../data/energy_consumption/DUQ_hourly.csv', '../data/energy_consumption/PJMW_hourly.csv']
# Shape: (178262, 13)
# Columns: ['Datetime', 'AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE', 'NI', 'PJME', 'PJMW', 'PJM_Load']
# 
# Missing values:
#  Datetime    0
# AEP         0
# COMED       0
# DAYTON      0
# DEOK        0
# DOM         0
# DUQ         0
# EKPC        0
# FE          0
# NI          0
# PJME        0
# PJMW        0
# PJM_Load    0
# dtype: int64
#              Datetime      AEP   COMED  DAYTON    DEOK     DOM     DUQ  \
# 0 1998-04-01 01:00:00  12379.0  9631.0  1621.0  2533.0  7190.0  1364.0   
# 1 1998-04-01 02:00:00  12379.0  9631.0  1621.0  2533.0  7190.0  1364.0   
# 2 1998-04-01 03:00:00  12379.0  9631.0  1621.0  2533.0  7190.0  1364.0   
# 3 1998-04-01 04:00:00  12379.0  9631.0  1621.0  2533.0  7190.0  1364.0   
# 4 1998-04-01 05:00:00  12379.0  9631.0  1621.0  2533.0  7190.0  1364.0   
# 
#      EKPC   FE      NI     PJME    PJMW  PJM_Load  
# 0  1166.0  0.0  9198.0  30393.0  4374.0   22259.0  
# 1  1166.0  0.0  9198.0  30393.0  4374.0   21244.0  
# 2  1166.0  0.0  9198.0  30393.0  4374.0   20651.0  
# 3  1166.0  0.0  9198.0  30393.0  4374.0   20421.0  
# 4  1166.0  0.0  9198.0  30393.0  4374.0   20713.0  
# --------------

# %% [cell 3]
# Detect target (energy/MW/consumption)
energy_cols = [c for c in df.columns if any(k in c.lower()
               for k in ['mw','energy','consumption','kwh','load','demand'])]
TARGET_COL = energy_cols[0] if energy_cols else df.select_dtypes(include=np.number).columns[0]
print(f'Target column: {TARGET_COL}')

# Parse datetime if available
date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date','time','datetime'])]
if date_cols:
    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    df = df.dropna(subset=[date_cols[0]]).sort_values(date_cols[0]).reset_index(drop=True)

df = df.dropna(subset=[TARGET_COL])
data_series = df[TARGET_COL].values.astype(float)
print(f'Total time steps: {len(data_series):,}')

# Limit to last 20000 for speed
if len(data_series) > 20000:
    data_series = data_series[-20000:]
    print(f'Trimmed to last 20,000 steps')

# --- OUTPUT ---
# Target column: PJMW
# Total time steps: 178,262
# Trimmed to last 20,000 steps
# --------------

# %% [cell 4]
fig, axes = plt.subplots(3, 1, figsize=(15, 11))

# Full series
axes[0].plot(data_series, color='#2196F3', lw=0.8, alpha=0.9)
axes[0].fill_between(range(len(data_series)), data_series,
                     data_series.min(), alpha=0.15, color='#2196F3')
axes[0].set_title(f'{TARGET_COL} - Toàn bộ Chuỗi Thời gian')
axes[0].set_ylabel('Energy (MW)')

# Rolling stats
s = pd.Series(data_series)
roll_mean = s.rolling(window=168).mean()  # 7-day (168 hrs)
roll_std  = s.rolling(window=168).std()

axes[1].plot(data_series, color='#2196F3', lw=0.5, alpha=0.5, label='Raw')
axes[1].plot(roll_mean, color='#FF9800', lw=2, label='7-Day Rolling Mean')
axes[1].fill_between(range(len(data_series)),
                     (roll_mean + 2*roll_std).values,
                     (roll_mean - 2*roll_std).values,
                     alpha=0.15, color='#FF9800', label='±2σ Band')
axes[1].set_title('Rolling Mean ± 2σ (7-Day Window)')
axes[1].legend(fontsize=9)

# Zoom: 1 week
show = min(168, len(data_series))
axes[2].plot(data_series[:show], 'o-', color='#E91E63', lw=1.5, ms=3)
axes[2].fill_between(range(show), data_series[:show],
                     data_series[:show].min(), alpha=0.2, color='#E91E63')
axes[2].set_title(f'Zoom: Đầu tiên {show} bước (~1 tuần)')
axes[2].set_xlabel('Time Step')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/01_energy_eda.png', bbox_inches='tight')
plt.show()

print(f'Mean: {data_series.mean():.2f} | Std: {data_series.std():.2f} | '
      f'Min: {data_series.min():.2f} | Max: {data_series.max():.2f}')

# --- OUTPUT ---
# <Figure size 1800x1320 with 3 Axes>Mean: 5583.61 | Std: 1001.56 | Min: 3420.00 | Max: 9342.00
# --------------

# %% [cell 5]
# 1. Distribution of Energy Consumption
plt.figure(figsize=(7.5, 4.5))

mean_val = np.mean(data_series)
plt.hist(data_series, bins=60, color='#2196F3', edgecolor='white', alpha=0.85, label='Energy Distribution')
plt.axvline(mean_val, color='red', linestyle='--', lw=2, label=f'Mean = {mean_val:.2f}')

plt.title('Phân phối Energy Consumption', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Energy (MW or kWh)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()

# Lưu biểu đồ phân phối tiêu thụ năng lượng
plt.savefig(f'{SAVE_DIR}/02_a_energy_distribution.png', dpi=120, bbox_inches='tight')
plt.show()

# --- OUTPUT ---
# <Figure size 900x540 with 1 Axes>
# --------------

# %% [cell 6]
# 2. Autocorrelation Analysis
plt.figure(figsize=(8.5, 4.5))

max_lag = 100
autocorr = [pd.Series(data_series).autocorr(lag=l) for l in range(1, max_lag + 1)]

plt.bar(range(1, max_lag + 1), autocorr, color='steelblue', alpha=0.8, label='Autocorrelation')
plt.axhline(0, color='black', lw=1)
plt.axhline(0.05, color='red', linestyle='--', lw=1.5, label='±0.05 Confidence Interval')
plt.axhline(-0.05, color='red', linestyle='--', lw=1.5)

plt.title('Autocorrelation (lag 1–100)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation Coefficient')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()

# Lưu biểu đồ tự tương quan
plt.savefig(f'{SAVE_DIR}/02_b_autocorrelation.png', dpi=120, bbox_inches='tight')
plt.show()

# --- OUTPUT ---
# <Figure size 1020x540 with 1 Axes>
# --------------

# %% [cell 7]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_series.reshape(-1,1)).flatten()

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, LOOK_BACK)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'X: {X.shape} → Train: {X_train.shape}, Test: {X_test.shape}')

# --- OUTPUT ---
# X: (19952, 48, 1) → Train: (15961, 48, 1), Test: (3991, 48, 1)
# --------------

# %% [cell 8]
# Custom Bahdanau-style Attention Layer
class AttentionLayer(layers.Layer):
    """Cơ chế Chú ý (Attention) tự xây dựng.
    
    Mô hình học cách phân bổ trọng số (attention weights) cho
    từng bước thời gian trong chuỗi đầu vào.
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V  = layers.Dense(1)

    def call(self, features):
        # features: (batch, timesteps, hidden_dim)
        score = tf.nn.tanh(self.W1(features) + self.W2(features))
        # score: (batch, timesteps, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # attention_weights: (batch, timesteps, 1)
        context = attention_weights * features
        context = tf.reduce_sum(context, axis=1)
        # context: (batch, hidden_dim)
        return context, attention_weights

print('Custom Attention Layer defined')

# --- OUTPUT ---
# Custom Attention Layer defined
# --------------

# %% [cell 9]
# Build with Functional API to expose attention weights
inp = tf.keras.Input(shape=(LOOK_BACK, 1), name='input')

x = layers.LSTM(128, return_sequences=True, name='lstm_1')(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(64, return_sequences=True, name='lstm_2')(x)
x = layers.Dropout(0.3)(x)

# Apply Attention
context, attn_weights = AttentionLayer(units=64, name='attention')(x)

x = layers.Dense(32, activation='relu')(context)
out = layers.Dense(1, name='forecast')(x)

model = tf.keras.Model(inputs=inp, outputs=out, name='LSTM_Attention_Energy')

# Separate model to extract attention weights during inference
attn_model = tf.keras.Model(inputs=inp, outputs=attn_weights, name='AttentionExtractor')

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()

# --- OUTPUT ---
# RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY: out of memory
# --------------

# %% [cell 10]
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ],
    verbose=1
)

# --- OUTPUT ---
# Epoch 1/40
# 225/225 [==============================] - 9s 33ms/step - loss: 0.0126 - mae: 0.0835 - val_loss: 0.0802 - val_mae: 0.2488 - lr: 0.0010
# Epoch 2/40
# 225/225 [==============================] - 7s 29ms/step - loss: 0.0060 - mae: 0.0609 - val_loss: 0.0326 - val_mae: 0.1514 - lr: 0.0010
# Epoch 3/40
# 225/225 [==============================] - 7s 30ms/step - loss: 0.0050 - mae: 0.0555 - val_loss: 0.0306 - val_mae: 0.1429 - lr: 0.0010
# Epoch 4/40
# 225/225 [==============================] - 7s 31ms/step - loss: 0.0045 - mae: 0.0527 - val_loss: 0.0140 - val_mae: 0.0973 - lr: 0.0010
# Epoch 5/40
# 225/225 [==============================] - 7s 30ms/step - loss: 0.0038 - mae: 0.0483 - val_loss: 0.0063 - val_mae: 0.0638 - lr: 0.0010
# Epoch 6/40
# 225/225 [==============================] - 7s 30ms/step - loss: 0.0033 - mae: 0.0448 - val_loss: 0.0079 - val_mae: 0.0712 - lr: 0.0010
# Epoch 7/40
# 225/225 [==============================] - 7s 30ms/step - loss: 0.0026 - mae: 0.0398 - val_loss: 0.0057 - val_mae: 0.0616 - lr: 0.0010
# Epoch 8/40
# 225/225 [==============================] - 7s 31ms/step - loss: 0.0018 - mae: 0.0335 - val_loss: 0.0014 - val_mae: 0.0300 - lr: 0.0010
# Epoch 9/40
# 225/225 [==============================] - 7s 31ms/step - loss: 0.0017 - mae: 0.0324 - val_loss: 0.0012 - val_mae: 0.0274 - lr: 0.0010
# Epoch 10/40
# 225/225 [==============================] - 7s 30ms/step - loss: 0.0016 - mae: 0.0318 - val_loss: 0.0011 - val_mae: 0.0265 - lr: 0.0010
# Epoch 11/40
# 225/225 [==============================] - 7s 31ms/step - loss: 0.0013 - mae: 0.0281 - val_loss: 0.0014 - val_mae: 0.0305 - lr: 0.0010
# Epoch 12/40
# 225/225 [==============================] - 7s 29ms/step - loss: 0.0013 - mae: 0.0278 - val_loss: 9.4813e-04 - val_mae: 0.0243 - lr: 0.0010
# Epoch 13/40
# 225/225 [==============================] - 7s 31ms/step - loss: 0.0011 - mae: 0.0259 - val_loss: 8.3723e-04 - val_mae: 0.0229 - lr: 0.0010
# Epoch 14/40
# 225/225 [==============================] - 7s 30ms/step - loss: 0.0010 - mae: 0.0252 - val_loss: 6.3705e-04 - val_mae: 0.0196 - lr: 0.0010
# Epoch 15/40
# 225/225 [==============================] - 7s 30ms/step - loss: 9.4772e-04 - mae: 0.0242 - val_loss: 8.4964e-04 - val_mae: 0.0225 - lr: 0.0010
# Epoch 16/40
# 225/225 [==============================] - 7s 31ms/step - loss: 9.3066e-04 - mae: 0.0240 - val_loss: 6.7763e-04 - val_mae: 0.0206 - lr: 0.0010
# Epoch 17/40
# 225/225 [==============================] - ETA: 0s - loss: 9.3131e-04 - mae: 0.0238
# Epoch 17: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 225/225 [==============================] - 7s 29ms/step - loss: 9.3131e-04 - mae: 0.0238 - val_loss: 6.2404e-04 - val_mae: 0.0200 - lr: 0.0010
# Epoch 18/40
# 225/225 [==============================] - 7s 31ms/step - loss: 8.2012e-04 - mae: 0.0225 - val_loss: 7.9064e-04 - val_mae: 0.0234 - lr: 5.0000e-04
# Epoch 19/40
# 225/225 [==============================] - 7s 31ms/step - loss: 8.3646e-04 - mae: 0.0227 - val_loss: 7.7675e-04 - val_mae: 0.0227 - lr: 5.0000e-04
# Epoch 20/40
# 225/225 [==============================] - 7s 30ms/step - loss: 7.7478e-04 - mae: 0.0219 - val_loss: 5.2751e-04 - val_mae: 0.0188 - lr: 5.0000e-04
# Epoch 21/40
# 225/225 [==============================] - 7s 31ms/step - loss: 8.0721e-04 - mae: 0.0223 - val_loss: 9.6866e-04 - val_mae: 0.0263 - lr: 5.0000e-04
# Epoch 22/40
# 225/225 [==============================] - 7s 30ms/step - loss: 8.1564e-04 - mae: 0.0224 - val_loss: 0.0013 - val_mae: 0.0293 - lr: 5.0000e-04
# Epoch 23/40
# 223/225 [============================>.] - ETA: 0s - loss: 7.8609e-04 - mae: 0.0217
# Epoch 23: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
# 225/225 [==============================] - 7s 31ms/step - loss: 7.8765e-04 - mae: 0.0217 - val_loss: 0.0018 - val_mae: 0.0369 - lr: 5.0000e-04
# Epoch 24/40
# 225/225 [==============================] - 7s 31ms/step - loss: 7.5349e-04 - mae: 0.0215 - val_loss: 3.8550e-04 - val_mae: 0.0150 - lr: 2.5000e-04
# Epoch 25/40
# 225/225 [==============================] - 7s 30ms/step - loss: 6.8882e-04 - mae: 0.0204 - val_loss: 6.1210e-04 - val_mae: 0.0199 - lr: 2.5000e-04
# Epoch 26/40
# 225/225 [==============================] - 7s 30ms/step - loss: 7.0122e-04 - mae: 0.0208 - val_loss: 0.0014 - val_mae: 0.0333 - lr: 2.5000e-04
# Epoch 27/40
# 224/225 [============================>.] - ETA: 0s - loss: 6.6199e-04 - mae: 0.0201
# Epoch 27: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
# 225/225 [==============================] - 7s 31ms/step - loss: 6.6694e-04 - mae: 0.0202 - val_loss: 3.9242e-04 - val_mae: 0.0154 - lr: 2.5000e-04
# Epoch 28/40
# 225/225 [==============================] - 7s 31ms/step - loss: 6.6885e-04 - mae: 0.0202 - val_loss: 3.5800e-04 - val_mae: 0.0146 - lr: 1.2500e-04
# Epoch 29/40
# 225/225 [==============================] - 7s 30ms/step - loss: 6.4042e-04 - mae: 0.0199 - val_loss: 5.6408e-04 - val_mae: 0.0190 - lr: 1.2500e-04
# Epoch 30/40
# 224/225 [============================>.] - ETA: 0s - loss: 6.6116e-04 - mae: 0.0201
# Epoch 30: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
# 225/225 [==============================] - 7s 29ms/step - loss: 6.6485e-04 - mae: 0.0201 - val_loss: 3.5254e-04 - val_mae: 0.0147 - lr: 1.2500e-04
# Epoch 31/40
# 225/225 [==============================] - 7s 31ms/step - loss: 6.2745e-04 - mae: 0.0196 - val_loss: 3.7425e-04 - val_mae: 0.0153 - lr: 6.2500e-05
# Epoch 32/40
# 225/225 [==============================] - 7s 29ms/step - loss: 6.1930e-04 - mae: 0.0196 - val_loss: 4.0143e-04 - val_mae: 0.0158 - lr: 6.2500e-05
# Epoch 33/40
# 223/225 [============================>.] - ETA: 0s - loss: 6.4225e-04 - mae: 0.0199
# Epoch 33: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
# 225/225 [==============================] - 7s 30ms/step - loss: 6.4239e-04 - mae: 0.0199 - val_loss: 3.3232e-04 - val_mae: 0.0141 - lr: 6.2500e-05
# Epoch 34/40
# 225/225 [==============================] - 7s 31ms/step - loss: 6.0967e-04 - mae: 0.0195 - val_loss: 3.2649e-04 - val_mae: 0.0141 - lr: 3.1250e-05
# Epoch 35/40
# 225/225 [==============================] - 7s 29ms/step - loss: 6.4263e-04 - mae: 0.0199 - val_loss: 3.3927e-04 - val_mae: 0.0144 - lr: 3.1250e-05
# Epoch 36/40
# 224/225 [============================>.] - ETA: 0s - loss: 6.1824e-04 - mae: 0.0195
# Epoch 36: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
# 225/225 [==============================] - 7s 31ms/step - loss: 6.2157e-04 - mae: 0.0195 - val_loss: 3.5991e-04 - val_mae: 0.0148 - lr: 3.1250e-05
# Epoch 37/40
# 225/225 [==============================] - 7s 30ms/step - loss: 5.9223e-04 - mae: 0.0191 - val_loss: 3.3222e-04 - val_mae: 0.0142 - lr: 1.5625e-05
# Epoch 38/40
# 225/225 [==============================] - 7s 30ms/step - loss: 6.3651e-04 - mae: 0.0198 - val_loss: 3.4164e-04 - val_mae: 0.0145 - lr: 1.5625e-05
# Epoch 39/40
# 224/225 [============================>.] - ETA: 0s - loss: 6.2336e-04 - mae: 0.0196
# Epoch 39: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.
# 225/225 [==============================] - 7s 31ms/step - loss: 6.2488e-04 - mae: 0.0197 - val_loss: 3.3681e-04 - val_mae: 0.0142 - lr: 1.5625e-05
# Epoch 40/40
# 225/225 [==============================] - 7s 29ms/step - loss: 6.1495e-04 - mae: 0.0195 - val_loss: 3.3172e-04 - val_mae: 0.0142 - lr: 7.8125e-06
# --------------

# %% [cell 11]
y_pred_s    = model.predict(X_test, verbose=0).flatten()
y_pred_orig = scaler.inverse_transform(y_pred_s.reshape(-1,1)).flatten()
y_test_orig = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae  = mean_absolute_error(y_test_orig, y_pred_orig)
r2   = r2_score(y_test_orig, y_pred_orig)

print(f'RMSE : {rmse:.4f} MW')
print(f'MAE  : {mae:.4f} MW')
print(f'R²   : {r2:.4f}')

# --- OUTPUT ---
# RMSE : 94.1329 MW
# MAE  : 74.4881 MW
# R²   : 0.9884
# --------------

# %% [cell 12]
fig, axes = plt.subplots(2, 1, figsize=(15, 9))

# Forecast
show_n = min(300, len(y_test_orig))
axes[0].plot(y_test_orig[:show_n], color='steelblue', lw=2, label='Actual')
axes[0].plot(y_pred_orig[:show_n], color='#FF5722', lw=2, linestyle='--', label='Predicted')
axes[0].fill_between(range(show_n), y_test_orig[:show_n], y_pred_orig[:show_n],
                     alpha=0.2, color='gray')
axes[0].set_title(f'Dự báo Energy Consumption - RMSE={rmse:.2f}, R²={r2:.4f}')
axes[0].set_ylabel('Energy (MW)')
axes[0].legend()

# Error distribution
errors = y_test_orig - y_pred_orig
axes[1].hist(errors, bins=60, color='darkorange', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='black', linestyle='--', lw=2)
axes[1].axvline(errors.mean(), color='red', linestyle=':', lw=2,
                label=f'Mean={errors.mean():.2f}')
axes[1].set_title('Error Distribution - Error = Actual - Predicted')
axes[1].set_xlabel('Error (MW)')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/03_forecast_error.png', bbox_inches='tight')
plt.show()

# --- OUTPUT ---
# <Figure size 1800x1080 with 2 Axes>
# --------------

# %% [cell 13]
# Attention Weight Heatmap
n_samples_viz = 20
sample_X  = X_test[:n_samples_viz]
attn_vals = attn_model.predict(sample_X, verbose=0)  # (n_samples, look_back, 1)
attn_vals = attn_vals.squeeze(-1)  # (n_samples, look_back)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(attn_vals, cmap='YlOrRd', ax=ax,
            xticklabels=[f't-{LOOK_BACK-i}' if i % 8 == 0 else ''
                         for i in range(LOOK_BACK)],
            yticklabels=[f'Sample {i+1}' for i in range(n_samples_viz)],
            linewidths=0.3)
ax.set_title('Attention Weight Heatmap',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Timestep (t-48 → t-1)')
ax.set_ylabel('Test Sample')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/04_attention_heatmap.png', bbox_inches='tight')
plt.show()
# print('\nMỗi hàng = 1 sample dự báo, màu càng đậm = Attention càng cao.')
# print('Mô hình tập trung nhiều nhất vào các bước thời gian gần nhất (t-1, t-2,...)')

# --- OUTPUT ---
# <Figure size 1680x720 with 2 Axes>
# Mỗi hàng = 1 sample dự báo, màu càng đậm = Attention càng cao.
# Mô hình tập trung nhiều nhất vào các bước thời gian gần nhất (t-1, t-2,...)
# --------------

# %% [cell 14]
# Average attention over all test samples
all_attn = attn_model.predict(X_test, verbose=0).squeeze(-1)  # (N, look_back)
mean_attn = all_attn.mean(axis=0)

fig, ax = plt.subplots(figsize=(12, 4))
steps = [f't-{LOOK_BACK-i}' for i in range(LOOK_BACK)]
ax.bar(range(LOOK_BACK), mean_attn,
       color=plt.cm.YlOrRd(mean_attn / mean_attn.max()),
       edgecolor='black', linewidth=0.3)
ax.set_xticks(range(0, LOOK_BACK, 6))
ax.set_xticklabels([steps[i] for i in range(0, LOOK_BACK, 6)], rotation=30)
ax.set_title('Attention Weight Mean - Tầm quan trọng theo Timestep', fontweight='bold')
ax.set_xlabel('Timestep')
ax.set_ylabel('Mean Attention Weight')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/05_mean_attention.png', bbox_inches='tight')
plt.show()

# --- OUTPUT ---
# <Figure size 1440x480 with 1 Axes>
# --------------

# %% [cell 15]
with open(f'{SAVE_DIR}/report.txt', 'w') as f:
    f.write('Energy Consumption - LSTM + Attention\n' + '='*50 + '\n')
    f.write(f'LOOK_BACK : {LOOK_BACK} steps\n')
    f.write(f'Params    : {model.count_params():,}\n\n')
    f.write(f'Test RMSE : {rmse:.4f} MW\n')
    f.write(f'Test MAE  : {mae:.4f} MW\n')
    f.write(f'Test R²   : {r2:.4f}\n')

print('Energy LSTM+Attention Done! Saved to', SAVE_DIR)

# --- OUTPUT ---
# ✅ Energy LSTM+Attention Done! Saved to ../results/energy_attention
# --------------


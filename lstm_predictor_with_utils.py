
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Model Definition
class LSTMHybrid(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_reg = nn.Linear(hidden_size, 1)
        self.fc_cls = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc_reg(out), self.sigmoid(self.fc_cls(out))

# Prediction on entire dataset
def predict_new_data(new_df, model, feature_scaler, target_scaler, features, seq_len=5):
    new_df = new_df.sort_values(by=['CHUTE', 'ZIP_CODE', 'HOUR_TIME']).reset_index(drop=True)
    new_df['HOUR_TIME'] = pd.to_datetime(new_df['HOUR_TIME'])
    new_df['CHUTE'] = new_df['CHUTE'].astype('category').cat.codes
    new_df['ZIP_CODE'] = new_df['ZIP_CODE'].astype('category').cat.codes
    new_df[features] = feature_scaler.transform(new_df[features])

    data = new_df[features].values
    X_new = [data[i:i+seq_len] for i in range(len(data) - seq_len)]
    X_new_tensor = torch.tensor(np.array(X_new), dtype=torch.float32)

    with torch.no_grad():
        y_pred_reg, y_pred_cls = model(X_new_tensor)
        predicted_time = target_scaler.inverse_transform(y_pred_reg.view(-1).reshape(-1, 1)).flatten()
        predicted_flag = (y_pred_cls.view(-1).numpy() > 0.5).astype(int)

    result_df = pd.DataFrame({
        "Predicted_Processing_Time": predicted_time,
        "Predicted_Performance_Issue": predicted_flag
    })
    return result_df

# Prediction at a specific time
def predict_at_time(df_raw, target_time, model, feature_scaler, target_scaler, features, seq_len=5):
    df_raw['HOUR_TIME'] = pd.to_datetime(df_raw['HOUR_TIME'])
    df = df_raw[df_raw['HOUR_TIME'] <= pd.to_datetime(target_time)]
    df = df.sort_values(by='HOUR_TIME').reset_index(drop=True)

    if len(df) < seq_len:
        print("❌ Not enough history before the given time.")
        return None

    seq_df = df.iloc[-seq_len:].copy()
    seq_df['CHUTE'] = seq_df['CHUTE'].astype('category').cat.codes
    seq_df['ZIP_CODE'] = seq_df['ZIP_CODE'].astype('category').cat.codes
    seq_df[features] = feature_scaler.transform(seq_df[features])

    x_seq = torch.tensor([seq_df[features].values], dtype=torch.float32)

    with torch.no_grad():
        y_pred_reg, y_pred_cls = model(x_seq)
        predicted_time = target_scaler.inverse_transform(y_pred_reg.view(-1).reshape(-1, 1)).flatten()[0]
        predicted_issue = int(y_pred_cls.view(-1).item() > 0.5)

    print(f"📅 Time: {target_time}")
    print(f"⏱️ Predicted Avg Processing Time: {predicted_time:.2f} minutes")
    print(f"🚨 Performance Issue: {'YES' if predicted_issue else 'NO'}")

    return predicted_time, predicted_issue

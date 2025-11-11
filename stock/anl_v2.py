import numpy as np
import pandas as pd
import warnings
import os
import json
from datetime import datetime

# --- 導入 Scikit-learn 元件 ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge # 【新】導入 Ridge，作為我們的元模型

# --- 導入其他模型 ---
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# --- 導入 PyTorch (MLP) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 導入 Keras (LSTM) ---
from tensorflow.keras.models import Sequential, Model # 【新】導入 Model
from tensorflow.keras.layers import (
    LSTM, Dropout, Dense, Bidirectional, BatchNormalization, 
    Input, Attention, Concatenate # 【新】導入 Functional API 和 Attention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# --- 特徵定義 ---
TARGET = 'daily_return'
FEATURES_FILE = 'final_features.json'
HYPERPARAMS_FILE = 'best_hyperparameters.json' # 【新】指定超參數檔案
FEATURES = [] 
BEST_PARAMS = {} # 【新】用於存放載入的超參數

# --- 數據加載與預處理 ---
def load_and_prepare_data(file_path, features_file):
    global FEATURES 
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到數據文件: {file_path}")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"找不到特徵文件: {features_file}")

    # 1. 載入特徵列表
    try:
        with open(features_file, 'r') as f:
            FEATURES = json.load(f)
        print(f"成功從 {features_file} 載入 {len(FEATURES)} 個特徵。")
    except Exception as e:
        raise IOError(f"讀取特徵文件 {features_file} 失敗: {e}")
        
    # 2. 載入數據
    print(f"成功讀取數據文件: {file_path}")
    df = pd.read_csv(file_path)
    
    # 3. 數據清洗與排序
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 4. 執行與特徵工程腳本 *完全相同* 的特徵創建步驟
    df['Price_vs_EMA20'] = (df['Close'] - df['EMA20']) / df['Close']
    df['MACD_Line_Diff'] = (df['EMA12'] - df['EMA26']) / (df['Close'] + 1e-6) # 輕微優化：標準化
    df['RSI14_Diff'] = df['RSI14'].diff()
    df['Aroon_Oscillator_25'] = df['AROON_bull_25'] - df['AROON_bear_25']
    df['BB_Percent_B'] = (df['Close'] - df['LB']) / (df['UB'] - df['LB'] + 1e-6)
    df['Volatility_Ratio'] = df['STD_20'] / (df['SMA_20'] + 1e-6)
    df['High_Low_Range'] = (df['High'] - df['Low']) / (df['Close'].shift(1) + 1e-6)
    df['Lagged_DJI_Return'] = df['DJI_Close'].pct_change().shift(1)
    df['Lagged_Volume'] = df['Volume'].shift(1)
    df['Lagged_Return_1d'] = df[TARGET].shift(1) 
    df['DayOfWeek'] = df.index.dayofweek
    df['RSI14_State'] = pd.cut(df['RSI14'], bins=[0, 30, 70, 100], 
                                labels=['Oversold', 'Neutral', 'Overbought'], right=False)
    df = pd.get_dummies(df, columns=['RSI14_State', 'DayOfWeek'], drop_first=True)
    
    # 5. 檢查特徵完整性
    required_cols = FEATURES + [TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"警告：數據中缺少以下特徵 (可能是 one-hot 編碼名稱不匹配): {missing}")
        FEATURES = [f for f in FEATURES if f in df.columns]
        print(f"已更新特徵列表，剩餘特徵數: {len(FEATURES)}")

    # 6. 處理缺失值
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    df_final = df[FEATURES + [TARGET]].copy()
    df_final.dropna(inplace=True)

    return df_final

# --- 序列模型工具函式 ---
def create_sequences(data, seq_length):
    X, y = [], []
    if not isinstance(data, np.ndarray): data = np.array(data)
        
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1]) 
        y.append(data[i, -1]) 
    return np.array(X), np.array(y)

# --- 模型定義 (MLP / Attention-LSTM) ---

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# 【新】Attention-LSTM 模型 (使用 Keras Functional API)
def create_attention_lstm(shape, lstm_units=128, dense_units=64):
    """
    創建一個帶有 Attention 機制的 Bidirectional LSTM 模型
    """
    inputs = Input(shape=(shape[1], shape[2])) # (seq_len, n_features)
    
    # 雙向 LSTM 層
    # return_sequences=True: 為了讓 Attention 層能看到所有時間步的輸出
    # return_state=True: 為了獲取最終的隱藏狀態 (h) 和單元狀態 (c)
    bi_lstm, f_h, f_c, b_h, b_c = Bidirectional(
        LSTM(lstm_units, return_sequences=True, return_state=True)
    )(inputs)
    
    bi_lstm = Dropout(0.3)(bi_lstm)
    bi_lstm = BatchNormalization()(bi_lstm)

    # Keras 的 Attention 層會計算 query 和 value 之間的相似度
    # 我們使用 bi_lstm 的輸出作為 query 和 value (self-attention)
    attention_out = Attention()([bi_lstm, bi_lstm])
    
    # Attention 層的輸出仍然是一個序列，我們需要將其壓縮
    # 我們使用第二個 LSTM 層 (return_sequences=False) 來讀取這個加權後的序列
    lstm_condensed = LSTM(dense_units, return_sequences=False)(attention_out)
    
    lstm_condensed = Dropout(0.3)(lstm_condensed)
    output = Dense(1)(lstm_condensed)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# --- 集成模型訓練器類別 (已升級為 Stacking) ---
class EnsembleStockPredictor:
    def __init__(self, features, target, sequence_length=60, test_size=0.2):
        self.features = features
        self.target = target
        self.seq_len = sequence_length
        self.test_size = test_size
        self.models = {}
        self.scalers = {}
        self.test_data = {} # 存放 (X_test, y_test)
        self.preds = {}     # 存放 T+1 的最終預測
        self.volatility = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 【新】基礎模型名稱，必須與 self.preds 和 self.test_data 的鍵(key)一致
        self.base_model_names = [
            'rf', 'xgb', 'lgbm', 'svr', 'mlp', 'lstm', 'arima_garch'
        ]
        
        # 【新】元模型 (Meta-Model)
        self.meta_model = Ridge(alpha=1.0) # 使用 Ridge 回歸以增加穩定性

    # 數據準備 (T -> T+1)
    def _prepare_data(self, df):
        X_data = df[self.features].copy()
        y_data = df[self.target].shift(-1)
        
        X = X_data.iloc[:-1]
        y = y_data.iloc[:-1]
        
        X_values = np.nan_to_num(X.values, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        
        split_idx = int(len(X_values) * (1 - self.test_size))
        
        X_train, X_test = X_values[:split_idx], X_values[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.latest_X_point = X.iloc[-1].values 
        
        return X_train, X_test, y_train, y_test

    # 樹模型 RF, XGBoost, lightGBM
    def _train_tree_model(self, model_name, X_train, X_test, y_train, y_test, params={}):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[model_name] = scaler
        
        if model_name == 'rf':
            model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        elif model_name == 'xgb':
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **params)
        elif model_name == 'lgbm':
            model = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, verbose=-1, **params)

        model.fit(X_train_scaled, y_train)
        self.models[model_name] = model
        self.test_data[model_name] = (X_test_scaled, y_test)
        
        latest_point_scaled = scaler.transform(self.latest_X_point.reshape(1, -1))
        self.preds[model_name] = model.predict(latest_point_scaled)[0]
        print(f" 	{model_name.upper()} 預測 (T+1): {self.preds[model_name]*100:.4f}%")
        
    # SVR
    def train_svr(self, X_train, X_test, y_train, y_test, params={}):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['svr'] = scaler
        
        svr = SVR(**params)
        svr.fit(X_train_scaled, y_train)
        
        self.models['svr'] = svr
        self.test_data['svr'] = (X_test_scaled, y_test)
        
        latest_point_scaled = scaler.transform(self.latest_X_point.reshape(1, -1))
        self.preds['svr'] = svr.predict(latest_point_scaled)[0]
        print(f" 	SVR 預測 (T+1): {self.preds['svr']*100:.4f}%")

    # 【升級】Attention-LSTM
    def train_lstm(self, df):
        try:
            X_data = df[self.features].values
            y_data = df[self.target].values
            X_data_num = np.nan_to_num(X_data, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X_data_num)
            self.scalers['lstm'] = scaler
            
            data = np.concatenate((X_scaled, y_data.reshape(-1, 1)), axis=1)
            X_seq, y_seq = create_sequences(data, self.seq_len)

            split_idx = int(len(X_seq) * (1 - self.test_size))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # --- 【升級點】 ---
            model = create_attention_lstm((X_train.shape[1], X_train.shape[2]))
            # --- 【升級點結束】 ---
            
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            
            callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]
            
            model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=0)
            
            self.models['lstm'] = model
            self.test_data['lstm'] = (X_test, y_test)
            
            latest_sequence_unscaled = X_data_num[-self.seq_len:]
            latest_sequence_scaled = scaler.transform(latest_sequence_unscaled)
            
            latest_sequence = latest_sequence_scaled.reshape(1, self.seq_len, X_scaled.shape[1])
            self.preds['lstm'] = model.predict(latest_sequence, verbose=0)[0][0]
            print(f" 	LSTM (w/ Attention) 預測 (T+1): {self.preds['lstm']*100:.4f}%")
        except Exception as e:
            print(f" 	LSTM 訓練失敗: {e}. 略過 LSTM。")
            self.preds['lstm'] = 0.0
            self.test_data['lstm'] = (None, None)

    # MLP (原 Transformer)
    def train_mlp(self, X_train, X_test, y_train, y_test, params={}):
        try:
            device = self.device
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['mlp'] = scaler

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            model = MLPModel(input_dim=X_train.shape[1]).to(device) 
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            model.train()
            num_epochs = 50 
            for epoch in range(num_epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            self.models['mlp'] = model
            self.test_data['mlp'] = (X_test_tensor, y_test)
            
            with torch.no_grad():
                latest_point_scaled = scaler.transform(self.latest_X_point.reshape(1, -1))
                latest_point_tensor = torch.tensor(latest_point_scaled, dtype=torch.float32).to(device)
                self.preds['mlp'] = model(latest_point_tensor).cpu().numpy()[0][0]
            print(f" 	MLP 預測 (T+1): {self.preds['mlp']*100:.4f}%")
        except Exception as e:
            print(f" 	MLP 訓練失敗: {e}. 略過 MLP。")
            self.preds['mlp'] = 0.0
            self.test_data['mlp'] = (None, None)

    # ARIMA + GARCH
    def train_arima_garch(self, df):
        target_series = df[self.target]
        arima_order = (5, 1, 0) 
        
        try:
            if len(target_series) < 2 * (arima_order[0] + arima_order[1] + arima_order[2] + 1):
                 print(" 	ARIMA 數據長度不足，使用 (1, 0, 0)")
                 arima_order = (1, 0, 0) 
            
            # --- 【修改點】: 為了 Stacking，我們需要 T-1 的訓練集和 T 的測試集 ---
            split_idx = int(len(target_series) * (1 - self.test_size))
            y_train = target_series.iloc[:split_idx]
            y_test = target_series.iloc[split_idx:]
            
            # 1. 訓練模型以獲取 T+1 的預測 (使用所有數據)
            model_arima_full = ARIMA(target_series, order=arima_order).fit() 
            resid_full = model_arima_full.resid
            model_garch_full = arch_model(resid_full, vol='Garch', p=1, q=1).fit(update_freq=0, disp='off')
            
            forecast_arima = model_arima_full.forecast(steps=1)
            forecast_garch = model_garch_full.forecast(horizon=1)
            next_return = forecast_arima.iloc[-1] + forecast_garch.mean.values[0, 0]
            
            self.models['arima_garch'] = (model_arima_full, model_garch_full)
            self.preds['arima_garch'] = next_return
            self.volatility = forecast_garch.variance.values[0, 0] 
            
            # 2. 訓練模型以獲取「驗證集」的預測 (用於 Stacking)
            model_arima_train = ARIMA(y_train, order=arima_order).fit()
            # 使用 .forecast() 而不是 .predict() 來獲取樣本外(out-of-sample)預測
            test_preds = model_arima_train.forecast(steps=len(y_test))
            
            self.test_data['arima_garch'] = (test_preds.values, y_test.values) # 儲存 (preds, true)
            
            print(f" 	ARIMA+GARCH 預測 (T+1): {next_return*100:.4f}% (波動率: {self.volatility*100:.4f}%)")

        except Exception as e:
            print(f" 	ARIMA 擬合失敗: {e}. 略過 ARIMA/GARCH。")
            self.preds['arima_garch'] = 0.0
            self.test_data['arima_garch'] = (None, None) # 標記失敗
    
    # 執行所有基礎模型訓練
    def run_all_base_models(self, df, params={}):
        print("\n--- 開始訓練所有基礎模型 (步驟 2/4) ---")
        
        # 1. 準備 T -> T+1 的數據
        X_train, X_test, y_train, y_test = self._prepare_data(df)
        
        # 2. 訓練所有 X-based 模型 (T -> T+1)
        self._train_tree_model('rf', X_train, X_test, y_train, y_test, params.get('rf', {}))
        self._train_tree_model('xgb', X_train, X_test, y_train, y_test, params.get('xgb', {}))
        self._train_tree_model('lgbm', X_train, X_test, y_train, y_test, params.get('lgbm', {}))
        self.train_svr(X_train, X_test, y_train, y_test, params.get('svr', {}))
        self.train_mlp(X_train, X_test, y_train, y_test)
        
        # 3. 訓練 LSTM (T-seq...T-1 -> T)
        self.train_lstm(df)

        # 4. 訓練 ARIMA/GARCH (Y-only)
        self.train_arima_garch(df)

    # 【新】步驟 3: 訓練元模型 (Stacking)
    def train_meta_model(self):
        print("\n--- 訓練 Stacking 元模型 (步驟 3/4) ---")
        
        meta_features_train = {}
        meta_y_train = None
        
        base_model_names_in_data = [] # 用於 meta_model 的 feature_names_in_
        
        for name in self.base_model_names:
            if name not in self.test_data or self.test_data[name][0] is None:
                print(f" 	警告：找不到 {name} 的驗證數據，將從元模型中跳過。")
                continue
                
            print(f" 	收集 {name} 的驗證集預測...")
            X_test, y_test = self.test_data[name]
            
            if meta_y_train is None:
                # 確保 y_test 是 T+1 的標籤 (ARIMA 和 LSTM 需要對齊)
                if name == 'lstm' or name == 'arima_garch':
                    meta_y_train = y_test
                else:
                    # T -> T+1 的 y_test
                    meta_y_train = y_test 
            
            # 獲取預測
            preds_val = None
            try:
                if name == 'lstm':
                    preds_val = self.models[name].predict(X_test, verbose=0).flatten()
                elif name == 'mlp':
                    preds_val = self.models[name](X_test).cpu().detach().numpy().flatten()
                elif name == 'arima_garch':
                    preds_val = X_test # X_test 已經是預測值了
                else: # rf, xgb, lgbm, svr
                    preds_val = self.models[name].predict(X_test)
            except Exception as e:
                print(f" 	錯誤：獲取 {name} 預測時失敗: {e}。跳過此模型。")
                continue

            # 對齊所有預測的長度
            if len(preds_val) != len(meta_y_train):
                print(f" 	警告：{name} 的預測長度 ({len(preds_val)}) 與 meta_y ({len(meta_y_train)}) 不符。正在嘗試對齊...")
                # 截取或填充 (通常是 LSTM/ARIMA 與 X-based 模型差幾筆)
                min_len = min(len(preds_val), len(meta_y_train))
                preds_val = preds_val[:min_len]
                meta_y_train = meta_y_train[:min_len]
                
            meta_features_train[name] = preds_val
            base_model_names_in_data.append(name) # 記錄成功加入的模型

        # 創建元特徵 DataFrame
        meta_features_df = pd.DataFrame(meta_features_train)
        
        # 確保 y 也被截斷
        if meta_y_train is None or len(meta_y_train) == 0:
            print("錯誤：沒有可用的 y_test 數據來訓練元模型。")
            return {}
            
        meta_y_train = meta_y_train[:len(meta_features_df)]

        # 訓練元模型
        self.meta_model.fit(meta_features_df, meta_y_train)
        
        # 獲取權重 (係數)
        weights = self.meta_model.coef_
        # 使用 base_model_names_in_data 來確保 zip 對齊
        final_weights = dict(zip(base_model_names_in_data, weights)) 
        
        print(f"\n✅ Stacking 元模型訓練完成。")
        print(f"   模型截距 (Intercept): {self.meta_model.intercept_:.6f}")
        print(f"   模型權重 (Coefficients):")
        for k, v in final_weights.items():
            print(f" 	- {k}: {v:.4f}")
            
        return final_weights

    # 【新】步驟 4: 使用元模型進行最終預測
    def predict_with_meta_model(self):
        print("\n--- 使用 Stacking 元模型進行最終預測 (步驟 4/4) ---")
        
        if not hasattr(self.meta_model, 'feature_names_in_'):
             print("錯誤：元模型尚未訓練，無法預測。")
             return 0.0

        # 1. 收集所有基礎模型對 T+1 的預測 (已在 self.preds 中)
        latest_preds_dict = {}
        for name in self.meta_model.feature_names_in_: # 確保順序正確
            if name in self.preds:
                latest_preds_dict[name] = self.preds[name]
            else:
                print(f"警告：{name} 沒有 T+1 預測值，將使用 0.0")
                latest_preds_dict[name] = 0.0
                
        latest_preds_df = pd.DataFrame([latest_preds_dict])
        
        # 2. 使用元模型預測
        final_prediction = self.meta_model.predict(latest_preds_df)
        
        return final_prediction[0]

# --- 主執行區 ---
if __name__ == '__main__':
    
    DATA_FILE = 'AAPL_historical_data.csv' 
    OUTPUT_FILE = 'aapl_prediction_result_v2_stacking.csv' 
    
    try:
        # 1. 載入超參數
        if os.path.exists(HYPERPARAMS_FILE):
            with open(HYPERPARAMS_FILE, 'r') as f:
                BEST_PARAMS = json.load(f)
            print(f"--- 成功載入超參數: {HYPERPARAMS_FILE} ---")
        else:
            print(f"--- 警告: 找不到 {HYPERPARAMS_FILE}, 將使用預設參數 ---")
            BEST_PARAMS = {
                'lgbm': { # 提供一個預設值，以防 'best_params.json' 不在
                    'subsample': 0.9, 'num_leaves': 31, 'n_estimators': 100, 
                    'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.6
                }
            }

        # 2. 數據加載與處理
        df_aapl = load_and_prepare_data(DATA_FILE, FEATURES_FILE)
        
        # 3. 初始化預測器
        predictor = EnsembleStockPredictor(features=FEATURES, target=TARGET, sequence_length=60)
        
        # 4. 步驟 2: 執行所有基礎模型訓練
        predictor.run_all_base_models(df_aapl, params=BEST_PARAMS) 

        # 5. 步驟 3: 訓練 Stacking 元模型
        final_weights = predictor.train_meta_model()
        
        # 6. 步驟 4: 執行最終預測
        final_pred = predictor.predict_with_meta_model()
        
        print(f"\n[最終結果] Stacking 集成模型預測 (T+1) 收益率: {final_pred*100:.4f}%")

        # 7. 將結果儲存為 CSV 文件
        print(f"\n--- 儲存結果至 CSV ---")
        
        results_data = {
            'Prediction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ensemble_Prediction_Return': [final_pred],
            'Vol_Prediction_Variance': [predictor.volatility]
        }
        
        if final_weights: # 確保 final_weights 不是空的
            for model_name, weight in final_weights.items():
                results_data[f'{model_name.upper()}_Weight'] = [weight]
            
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ 成功將結果儲存到 '{OUTPUT_FILE}' 文件中。")
        
    except FileNotFoundError as e:
        print(f"\n[執行失敗] 錯誤: {e}")
        print(f"請確保 '{DATA_FILE}', '{FEATURES_FILE}' 和 '{HYPERPARAMS_FILE}' 都在同一個目錄下。")
    except Exception as e:
        print(f"\n[執行失敗] 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc() # 打印詳細錯誤

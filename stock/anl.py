import numpy as np
import pandas as pd
import warnings
import os
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

#特徵
TARGET = 'daily_return'
#這邊是json免得他沒吃到
FEATURES_FILE = 'final_features.json'
FEATURES = []

#數據加載與預處理
def load_and_prepare_data(file_path, features_file):
    global FEATURES # 宣告修改全局變數
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到date{file_path}")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"找不到特徵{features_file}")

    # 1載入特徵列表
    try:
        with open(features_file, 'r') as f:
            FEATURES = json.load(f)
        print(f"where{features_file} 載{len(FEATURES)} 個特徵數")
    except Exception as e:
        raise IOError(f"讀取特{features_file} 失敗 {e}")
        
    # 2載入數據
    print(f"成功讀{file_path}")
    df = pd.read_csv(file_path)
    
    # 3數據清洗與排序
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 捕捉相對動量和位置特徵
    df['Price_vs_EMA20'] = (df['Close'] - df['EMA20']) / df['Close']
    df['MACD_Line_Diff'] = df['EMA12'] - df['EMA26']
    df['RSI14_Diff'] = df['RSI14'].diff()
    df['Aroon_Oscillator_25'] = df['AROON_bull_25'] - df['AROON_bear_25']
    df['BB_Percent_B'] = (df['Close'] - df['LB']) / (df['UB'] - df['LB'] + 1e-6)
    # 波動性強化特徵
    df['Volatility_Ratio'] = df['STD_20'] / (df['SMA_20'] + 1e-6)
    # 高低價差
    df['High_Low_Range'] = (df['High'] - df['Low']) / (df['Close'].shift(1) + 1e-6)
    # 納入滯後資訊
    df['Lagged_DJI_Return'] = df['DJI_Close'].pct_change().shift(1)
    df['Lagged_Volume'] = df['Volume'].shift(1)
    df['Lagged_Return_1d'] = df[TARGET].shift(1) # 確保這裡也有一樣的特徵
    # 分類特徵
    df['DayOfWeek'] = df.index.dayofweek
    df['RSI14_State'] = pd.cut(df['RSI14'], bins=[0, 30, 70, 100], 
                                labels=['Oversold', 'Neutral', 'Overbought'], right=False)
    # 讀熱
    df = pd.get_dummies(df, columns=['RSI14_State', 'DayOfWeek'], drop_first=True)
    
    # 檢查特徵完整
    required_cols = FEATURES + [TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"缺 (可能是 one-hot 編碼名稱不匹配): {missing}")
        # 更新 FEATURES 列表，只保留存在於 df 中的特徵
        FEATURES = [f for f in FEATURES if f in df.columns]
        print(f"已更新特徵列表，剩餘特徵數: {len(FEATURES)}")

    # 處理缺失值
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 在dropna之前，確保 FEATURES 都在
    df_final = df[FEATURES + [TARGET]].copy()
    
    # 刪除因滯後或 bfill/ffill 無法填充的初始行
    df_final.dropna(inplace=True)

    return df_final

# 序列模型
def create_sequences(data, seq_length):
    X, y = [], []
    if not isinstance(data, np.ndarray): data = np.array(data)
        
    for i in range(seq_length, len(data)):
        # 天數-1
        X.append(data[i-seq_length:i, :-1]) 
        # 測下一天
        y.append(data[i, -1]) 
    return np.array(X), np.array(y)

# TF
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

# 集成模型訓練器類別
class EnsembleStockPredictor:
    def __init__(self, features, target, sequence_length=60, test_size=0.2):
        self.features = features
        self.target = target
        self.seq_len = sequence_length
        self.test_size = test_size
        self.models = {}
        self.scalers = {}
        self.test_data = {}
        self.preds = {}
        self.volatility = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始權重
        self.aggressive_weights = {
            "mlp": 0.2, "arima_garch": 0.2, "svr": 0.05, 
            "xgb": 0.05, "lgbm": 0.2, "rf": 0.05, "lstm": 0.25
        }
        # 確保初始權重總和為 1
        total_sum = sum(self.aggressive_weights.values())
        self.aggressive_weights = {k: v / total_sum for k, v in self.aggressive_weights.items()}

    # 預測
    def _prepare_data(self, df):
        
        # X
        X_data = df[self.features].copy()
        # y未來
        y_data = df[self.target].shift(-1)
        
        # 由於 y 上移了一天，X 的最後一行沒有對應的 y，y 的最後一行變為 NaN
        # 我們需要刪除最後一行來對齊
        X = X_data.iloc[:-1]
        y = y_data.iloc[:-1]
        
        #房個無窮大
        X_values = np.nan_to_num(X.values, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        
        # 使用索引切分確保時序性 (shuffle=False)
        split_idx = int(len(X_values) * (1 - self.test_size))
        
        X_train, X_test = X_values[:split_idx], X_values[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 返回 X_test 的最後一個點（即 T 時刻的特徵），用於 T+1 的預測
        self.latest_X_point = X.iloc[-1].values 
        
        return X_train, X_test, y_train, y_test

    # 樹模型 RF, XGBoost, lightGBM
    def _train_tree_model(self, model_name, X_train, X_test, y_train, y_test, params={}):
        
        # 正規化我應該只用在非序列
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[model_name] = scaler
        
        if model_name == 'rf':
            #超參數
            model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=4, **params)
        elif model_name == 'xgb':
            # 超參數
            model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', eval_metric='rmse', n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42, **params)
        elif model_name == 'lgbm':
            # 超參數
            model = lgb.LGBMRegressor(
                objective='regression', 
                metric='rmse', 
                random_state=42, 
                **params 
            )

        model.fit(X_train_scaled, y_train)
        self.models[model_name] = model
        self.test_data[model_name] = (X_test_scaled, y_test)
        
        # 使用 T 時刻的特徵 (latest_X_point) 來預測 T+1
        latest_point_scaled = scaler.transform(self.latest_X_point.reshape(1, -1))
        self.preds[model_name] = model.predict(latest_point_scaled)[0]
        print(f" 	{model_name.upper()} 預測 (T+1): {self.preds[model_name]*100:.4f}%")
        
    # SVR
    def train_svr(self, X_train, X_test, y_train, y_test, params={}):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['svr'] = scaler
        
        svr = SVR(kernel='linear', C=0.1, gamma='scale', epsilon=0.01, **params)
        svr.fit(X_train_scaled, y_train)
        
        self.models['svr'] = svr
        self.test_data['svr'] = (X_test_scaled, y_test)
        
        # 使用 T 時刻的特徵 (latest_X_point) 來預測 T+1
        latest_point_scaled = scaler.transform(self.latest_X_point.reshape(1, -1))
        self.preds['svr'] = svr.predict(latest_point_scaled)[0]
        print(f" 	SVR 預測 (T+1): {self.preds['svr']*100:.4f}%")

    # LSTM T那邊搞過了
    def train_lstm(self, df):
        # 準備 LSTM 數據 (T-seq...T-1 -> T)
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
        
        # 定義 Keras LSTM 模型結構
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3), BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.3), Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        # 定義回調函數
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]
        
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=0)
        
        self.models['lstm'] = model
        self.test_data['lstm'] = (X_test, y_test)
        
        # 準備 T+1 預測所需的序列: 使用 T-seq+1 到 T 的數據
        latest_sequence_unscaled = X_data_num[-self.seq_len:]
        latest_sequence_scaled = scaler.transform(latest_sequence_unscaled)
        
        latest_sequence = latest_sequence_scaled.reshape(1, self.seq_len, X_scaled.shape[1])
        self.preds['lstm'] = model.predict(latest_sequence, verbose=0)[0][0]
        print(f" 	LSTM 預測 (T+1): {self.preds['lstm']*100:.4f}%")
        
    # MLP/TF
    def train_mlp(self, X_train, X_test, y_train, y_test, params={}):
        device = self.device
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['mlp'] = scaler

        # 轉換為 PyTorch 張量
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = MLPModel(input_dim=X_train.shape[1]).to(device) # 使用 MLPModel
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # PyTorch
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
        
        # 進行預測
        with torch.no_grad():
            # 使用 T 時刻的特徵 (latest_X_point) 來預測 T+1
            latest_point_scaled = scaler.transform(self.latest_X_point.reshape(1, -1))
            latest_point_tensor = torch.tensor(latest_point_scaled, dtype=torch.float32).to(device)
            self.preds['mlp'] = model(latest_point_tensor).cpu().numpy()[0][0]
        print(f" 	MLP (原 Transformer) 預測 (T+1): {self.preds['mlp']*100:.4f}%")

    # ARIMA + GARCH 僅看Y
    def train_arima_garch(self, df):
        # ARIMA/GARCH 預測 T+1，它只使用 T 之前的 y 數據，所以是安全的
        target_series = df[self.target]
        arima_order = (5, 1, 0) # 預設參數
        
        try:
            
            if len(target_series) < 2 * (arima_order[0] + arima_order[1] + arima_order[2] + 1):
                 print(" 	ARIMA 數據長度不足，使用較小參數 (1, 0, 0)")
                 arima_order = (1, 0, 0) # 降級參數
            
            model_arima = ARIMA(target_series, order=arima_order)
            model_fit_arima = model_arima.fit() 
        except Exception as e:
            print(f" 	ARIMA 擬合失敗: {e}. 略過 ARIMA/GARCH 預測。")
            self.preds['arima_garch'] = 0.0
            return

        resid = model_fit_arima.resid
        model_garch = arch_model(resid, vol='Garch', p=1, q=1) # 預設 GARCH(1,1)
        model_fit_garch = model_garch.fit(update_freq=0, disp='off') # 關閉輸出
        
        # 進行一步預測
        forecast_arima = model_fit_arima.forecast(steps=1)
        forecast_garch = model_fit_garch.forecast(horizon=1)
                
        # ARIMA 預測值
        arima_mean = forecast_arima.iloc[-1]
        
        # GARCH 均值 
        garch_mean = forecast_garch.mean.values[0, 0] 
        
        # GARCH 波動率/變異數
        volatility = forecast_garch.variance.values[0, 0] 
        
        next_return = arima_mean + garch_mean
        
        self.models['arima_garch'] = (model_fit_arima, model_fit_garch)
        self.preds['arima_garch'] = next_return
        self.volatility = volatility
        print(f" 	ARIMA+GARCH 預測 (T+1): {next_return*100:.4f}% (波動率: {volatility*100:.4f}%)")
    
    # 執行所有模型訓練
    def run_all_models(self, df, rf_params={}, xgb_params={}, lgbm_params={}, svr_params={}):
        print("\n--- 開始訓練所有模型 (步驟 2/4) ---")
        
        # 1. ARIMA/GARCH (使用 T 之前的數據)
        self.train_arima_garch(df)

        # 2. 準備 T -> T+1 的數據
        X_train, X_test, y_train, y_test = self._prepare_data(df)
        
        # 3. 訓練所有非序列模型 (T -> T+1)
        self._train_tree_model('rf', X_train, X_test, y_train, y_test, rf_params)
        self._train_tree_model('xgb', X_train, X_test, y_train, y_test, xgb_params)
        self._train_tree_model('lgbm', X_train, X_test, y_train, y_test, lgbm_params)
        self.train_svr(X_train, X_test, y_train, y_test, svr_params)
        self.train_mlp(X_train, X_test, y_train, y_test) # 使用 T -> T+1
        
        # 4. 訓練 LSTM (T-seq...T-1 -> T)
        # 注意：LSTM 預測的是 T+1，與其他模型對齊
        self.train_lstm(df)

    # 動態權重分配
    def calculate_ensemble_prediction(self):
        volatility = self.volatility
        weights = self.aggressive_weights.copy()
        
        def normalize_weights(w):
            total = sum(w.values())
            if total == 0:
                print("警告：所有模型權重為零。")
                return {k: 0 for k in w}
            return {k: v / total for k, v in w.items()}

        # 根據波動率調整權重 (激進策略)
        if volatility > 0.0028:
            print("高波動: 激進策略採用保守調整")
            weights = {k: v * 0.8 for k, v in weights.items()}
            weights["lstm"] += 0.05
            weights["rf"] += 0.05

        elif 0.0015 < volatility <= 0.0028:
            print("中等波動: 激進策略採用平衡調整")
            weights = {k: v * 0.9 for k, v in weights.items()}
            weights["xgb"] += 0.05
            weights["arima_garch"] += 0.05

        else:
            print("低波動: 激進策略採用激進調整")
            weights = {k: v * 1.2 for k, v in weights.items()}
            weights["mlp"] += 0.05
            weights["svr"] += 0.05
            
        final_weights = normalize_weights(weights)
        
        ensemble_pred = 0
        for model_name, weight in final_weights.items():
            if model_name in self.preds:
                ensemble_pred += weight * self.preds[model_name]
        
        print("\n=== 權重分配與集成結果 (步驟 3/4) ===")
        print(f"最終波動率: {volatility*100:.4f}%")
        print("最終權重:", {k: f"{v:.4f}" for k, v in final_weights.items()})
        print(f"集成模型預測 (T+1 激進策略) 收益率: {ensemble_pred*100:.4f}%")
        
        return ensemble_pred, final_weights

# 主區
if __name__ == '__main__':
    
    DATA_FILE = 'AAPL_historical_data.csv' 
    OUTPUT_FILE = 'aapl_prediction_result.csv' 
    
    # 使用之前優化得到的最佳參數作為預設值
    DEFAULT_LGBM_PARAMS = {
        'subsample': 0.9, 
        'num_leaves': 31, 
        'n_estimators': 100, 
        'max_depth': 5, 
        'learning_rate': 0.05, 
        'colsample_bytree': 0.6
    }
    
    try:
        # 數據加載與處理
        df_aapl = load_and_prepare_data(DATA_FILE, FEATURES_FILE)
        
        print("\n--- LightGBM 超參數設定 (步驟 1/4) ---")
        print(f"使用的預設超參數: {DEFAULT_LGBM_PARAMS}")

        # 執行集成模型訓練與預測
        # 使用全局變數 FEATURES
        predictor = EnsembleStockPredictor(features=FEATURES, target=TARGET, sequence_length=60)
        
        # 傳入預設的 LightGBM 參數
        predictor.run_all_models(df_aapl, lgbm_params=DEFAULT_LGBM_PARAMS) 

        # 計算最終集成預測
        final_pred, final_weights = predictor.calculate_ensemble_prediction()
        
        # 將結果儲存為 CSV 文件
        print(f"\n--- 儲存結果至 CSV (步驟 4/4) ---")
        
        # 準備結果數據
        results_data = {
            'Prediction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ensemble_Prediction_Return': [final_pred],
            'Vol_Prediction_Variance': [predictor.volatility]
        }
        
        # 將權重轉換為獨立的欄位
        for model_name, weight in final_weights.items():
            results_data[f'{model_name.upper()}_Weight'] = [weight]
            
        # 創建 DataFrame
        results_df = pd.DataFrame(results_data)
        
        # 儲存到 CSV
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"成功將結果儲存到 '{OUTPUT_FILE}' 文件中。")
        
    except FileNotFoundError as e:
        print(f"\n[執行失敗] 錯誤: {e}")
        print("請確保 'AAPL_historical_data.csv' 和 'final_features.json' (由 feature_engineering.py 生成) 都在同一個目錄下。")
    except Exception as e:
        print(f"\n[執行失敗] 發生未預期的錯誤: {e}")

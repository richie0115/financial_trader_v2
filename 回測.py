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
from sklearn.linear_model import Ridge # 導入 Ridge，作為我們的元模型

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
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import (
    LSTM, Dropout, Dense, Bidirectional, BatchNormalization, 
    Input, Attention, Concatenate 
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# --- 特徵定義 ---
TARGET = 'daily_return'
FEATURES_FILE = 'final_features.json'
HYPERPARAMS_FILE = 'best_hyperparameters.json' 
FEATURES = [] 
BEST_PARAMS = {} 

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
    df['MACD_Line_Diff'] = (df['EMA12'] - df['EMA26']) / (df['Close'] + 1e-6) 
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

def create_attention_lstm(shape, lstm_units=128, dense_units=64):
    inputs = Input(shape=(shape[1], shape[2])) 
    
    bi_lstm, f_h, f_c, b_h, b_c = Bidirectional(
        LSTM(lstm_units, return_sequences=True, return_state=True)
    )(inputs)
    
    bi_lstm = Dropout(0.3)(bi_lstm)
    bi_lstm = BatchNormalization()(bi_lstm)
    attention_out = Attention()([bi_lstm, bi_lstm])
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
        self.test_data = {} 
        self.preds = {}     
        self.volatility = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_model_names = [
            'rf', 'xgb', 'lgbm', 'svr', 'mlp', 'lstm', 'arima_garch'
        ]
        
        self.meta_model = Ridge(alpha=1.0) 
        
        # 【新】用於存放回測結果的屬性
        self.backtest_results = None

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

    # Attention-LSTM
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
            
            model = create_attention_lstm((X_train.shape[1], X_train.shape[2]))
            
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
            
            split_idx = int(len(target_series) * (1 - self.test_size))
            y_train = target_series.iloc[:split_idx]
            y_test = target_series.iloc[split_idx:]
            
            model_arima_full = ARIMA(target_series, order=arima_order).fit() 
            resid_full = model_arima_full.resid
            model_garch_full = arch_model(resid_full, vol='Garch', p=1, q=1).fit(update_freq=0, disp='off')
            
            forecast_arima = model_arima_full.forecast(steps=1)
            forecast_garch = model_garch_full.forecast(horizon=1)
            next_return = forecast_arima.iloc[-1] + forecast_garch.mean.values[0, 0]
            
            self.models['arima_garch'] = (model_arima_full, model_garch_full)
            self.preds['arima_garch'] = next_return
            self.volatility = forecast_garch.variance.values[0, 0] 
            
            model_arima_train = ARIMA(y_train, order=arima_order).fit()
            test_preds = model_arima_train.forecast(steps=len(y_test))
            
            self.test_data['arima_garch'] = (test_preds.values, y_test.values) 
            
            print(f" 	ARIMA+GARCH 預測 (T+1): {next_return*100:.4f}% (波動率: {self.volatility*100:.4f}%)")

        except Exception as e:
            print(f" 	ARIMA 擬合失敗: {e}. 略過 ARIMA/GARCH。")
            self.preds['arima_garch'] = 0.0
            self.test_data['arima_garch'] = (None, None) 
    
    # 執行所有基礎模型訓練
    def run_all_base_models(self, df, params={}):
        print("\n--- 開始訓練所有基礎模型 (步驟 2/4) ---")
        
        X_train, X_test, y_train, y_test = self._prepare_data(df)
        
        self._train_tree_model('rf', X_train, X_test, y_train, y_test, params.get('rf', {}))
        self._train_tree_model('xgb', X_train, X_test, y_train, y_test, params.get('xgb', {}))
        self._train_tree_model('lgbm', X_train, X_test, y_train, y_test, params.get('lgbm', {}))
        self.train_svr(X_train, X_test, y_train, y_test, params.get('svr', {}))
        self.train_mlp(X_train, X_test, y_train, y_test)
        self.train_lstm(df)
        self.train_arima_garch(df)

    # 步驟 3: 訓練元模型 (Stacking)
    def train_meta_model(self):
        print("\n--- 訓練 Stacking 元模型 (步驟 3/4) ---")
        
        meta_features_train = {}
        meta_y_train = None
        
        base_model_names_in_data = [] 
        
        for name in self.base_model_names:
            if name not in self.test_data or self.test_data[name][0] is None:
                print(f" 	警告：找不到 {name} 的驗證數據，將從元模型中跳過。")
                continue
                
            print(f" 	收集 {name} 的驗證集預測...")
            X_test, y_test = self.test_data[name]
            
            if meta_y_train is None:
                if name == 'lstm' or name == 'arima_garch':
                    meta_y_train = y_test
                else:
                    meta_y_train = y_test 
            
            preds_val = None
            try:
                if name == 'lstm':
                    preds_val = self.models[name].predict(X_test, verbose=0).flatten()
                elif name == 'mlp':
                    preds_val = self.models[name](X_test).cpu().detach().numpy().flatten()
                elif name == 'arima_garch':
                    preds_val = X_test 
                else: 
                    preds_val = self.models[name].predict(X_test)
            except Exception as e:
                print(f" 	錯誤：獲取 {name} 預測時失敗: {e}。跳過此模型。")
                continue

            if len(preds_val) != len(meta_y_train):
                print(f" 	警告：{name} 的預測長度 ({len(preds_val)}) 與 meta_y ({len(meta_y_train)}) 不符。正在嘗試對齊...")
                min_len = min(len(preds_val), len(meta_y_train))
                preds_val = preds_val[:min_len]
                meta_y_train = meta_y_train[:min_len]
                
            meta_features_train[name] = preds_val
            base_model_names_in_data.append(name) 

        meta_features_df = pd.DataFrame(meta_features_train)
        
        if meta_y_train is None or len(meta_y_train) == 0:
            print("錯誤：沒有可用的 y_test 數據來訓練元模型。")
            return {}
            
        meta_y_train_series = pd.Series(meta_y_train, index=meta_features_df.index) # 確保索引對齊
        meta_y_train_series = meta_y_train_series[:len(meta_features_df)] # 再次截斷

        # 【修改點】: 獲取 y_test 的索引 (日期)
        # 我們假設所有 y_test (除了被截斷的) 都有相同的索引
        try:
            # _prepare_data 中的 y_test 帶有索引
            _, _, _, y_test_non_seq = self._prepare_data(df_aapl) 
            test_dates_index = y_test_non_seq.index[:len(meta_features_df)]
        except Exception as e:
            print(f"警告：獲取日期索引失敗 ({e})。將使用範圍索引。")
            test_dates_index = meta_features_df.index

        self.meta_model.fit(meta_features_df, meta_y_train_series)
        
        weights = self.meta_model.coef_
        final_weights = dict(zip(base_model_names_in_data, weights)) 
        
        print(f"\n✅ Stacking 元模型訓練完成。")
        print(f"   模型截距 (Intercept): {self.meta_model.intercept_:.6f}")
        print(f"   模型權重 (Coefficients):")
        for k, v in final_weights.items():
            print(f" 	- {k}: {v:.4f}")

        # --- 【新】回測邏輯開始 ---
        print("\n✅ 正在生成回測結果...")
        # 獲取 L2 元模型對驗證集(meta_features_df)的預測
        l2_predictions = self.meta_model.predict(meta_features_df)
        
        # 創建回測結果 DataFrame
        results_df = pd.DataFrame({
            'Date': test_dates_index,
            'Actual_Return': meta_y_train_series.values,
            'Predicted_Return': l2_predictions
        })
        results_df.set_index('Date', inplace=True)
        
        # 將 L1 模型的預測也加入，方便分析
        l1_preds_df = meta_features_df.copy()
        l1_preds_df.index = test_dates_index
        
        self.backtest_results = pd.concat([results_df, l1_preds_df.add_prefix('L1_Pred_')], axis=1)
        # --- 【新】回測邏輯結束 ---
            
        return final_weights

    # 步驟 4: 使用元模型進行最終預測
    def predict_with_meta_model(self):
        print("\n--- 使用 Stacking 元模型進行最終預測 (步驟 4/4) ---")
        
        if not hasattr(self.meta_model, 'feature_names_in_'):
             print("錯誤：元模型尚未訓練，無法預測。")
             return 0.0

        latest_preds_dict = {}
        for name in self.meta_model.feature_names_in_: 
            if name in self.preds:
                latest_preds_dict[name] = self.preds[name]
            else:
                print(f"警告：{name} 沒有 T+1 預測值，將使用 0.0")
                latest_preds_dict[name] = 0.0
                
        latest_preds_df = pd.DataFrame([latest_preds_dict])
        
        final_prediction = self.meta_model.predict(latest_preds_df)
        
        return final_prediction[0]

# --- 主執行區 ---
if __name__ == '__main__':
    
    DATA_FILE = 'AAPL_historical_data.csv' 
    FORECAST_OUTPUT_FILE = 'aapl_prediction_result_v2_stacking.csv' 
    BACKTEST_OUTPUT_FILE = 'backtest_results_v2.csv' # 【新】回測結果檔案
    
    df_aapl = None # 【新】將 df_aapl 提升到 try 區塊的頂層
    
    try:
        # 1. 載入超參數
        if os.path.exists(HYPERPARAMS_FILE):
            with open(HYPERPARAMS_FILE, 'r') as f:
                BEST_PARAMS = json.load(f)
            print(f"--- 成功載入超參數: {HYPERPARAMS_FILE} ---")
        else:
            print(f"--- 警告: 找不到 {HYPERPARAMS_FILE}, 將使用預設參數 ---")
            BEST_PARAMS = {
                'lgbm': { 
                    'subsample': 0.9, 'num_leaves': 31, 'n_estimators': 100, 
                    'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.6
                }
            }

        # 2. 數據加載與處理
        df_aapl = load_and_prepare_data(DATA_FILE, FEATURES_FILE)
        
        # 3. 初始化預測器
        predictor = EnsembleStockPredictor(features=FEATURES, target=TARGET, sequence_length=60)
        
        # 4. 步驟 2: 執行所有基礎模型訓練
        # 【修改】傳入 df_aapl，這樣 train_meta_model 才能訪問到它
        predictor.run_all_base_models(df_aapl, params=BEST_PARAMS) 

        # 5. 步驟 3: 訓練 Stacking 元模型 (此函數現在會自動生成回測)
        final_weights = predictor.train_meta_model()
        
        # 6. 步驟 4: 執行最終預測 (T+1)
        final_pred = predictor.predict_with_meta_model()
        
        print(f"\n[最終結果] Stacking 集成模型預測 (T+1) 收益率: {final_pred*100:.4f}%")

        # 7. 將「T+1 預測」結果儲存為 CSV 文件
        print(f"\n--- 儲存 T+1 預測結果至 CSV ---")
        
        results_data = {
            'Prediction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ensemble_Prediction_Return': [final_pred],
            'Vol_Prediction_Variance': [predictor.volatility]
        }
        
        if final_weights: 
            for model_name, weight in final_weights.items():
                results_data[f'{model_name.upper()}_Weight'] = [weight]
            
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(FORECAST_OUTPUT_FILE, index=False)
        print(f"✅ 成功將 T+1 預測儲存到 '{FORECAST_OUTPUT_FILE}' 文件中。")

        # 8. 【新】將「回測結果」儲存為 CSV 文件
        if predictor.backtest_results is not None:
            print(f"\n--- 儲存回測結果至 CSV ---")
            print(predictor.backtest_results.head())
            predictor.backtest_results.to_csv(BACKTEST_OUTPUT_FILE)
            print(f"✅ 成功將回測結果儲存到 '{BACKTEST_OUTPUT_FILE}' 文件中。")
        else:
            print("警告：未生成回測結果。")
            
    except FileNotFoundError as e:
        print(f"\n[執行失敗] 錯誤: {e}")
        print(f"請確保 '{DATA_FILE}', '{FEATURES_FILE}' 和 '{HYPERPARAMS_FILE}' 都在同一個目錄下。")
    except Exception as e:
        print(f"\n[執行失敗] 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc() # 打印詳細錯誤

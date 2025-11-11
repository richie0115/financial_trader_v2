import pandas as pd
import numpy as np
import json
import os
import warnings
from time import time

# --- 核心 sklearn 元件 ---
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error

# --- 導入模型 ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# --- 參數設定 ---
DATA_FILE = 'AAPL_historical_data.csv' 
FEATURES_FILE = 'final_features.json'
TARGET = 'daily_return'
HYPERPARAMS_OUTPUT_FILE = 'best_hyperparameters.json' # 最終產出的參數檔案

# 交叉驗證設定
N_SPLITS = 5 # 5 折的時序交叉驗證
N_ITER = 50  # 每個模型跑 50 次隨機搜索 (可依據你的電腦效能調整)

# --- 1. 數據載入與準備 (與 model_prediction.py 幾乎相同) ---

def load_and_prepare_data_for_tuning(file_path, features_file):
    """
    載入數據、載入特徵、並執行與預測腳本相同的特徵工程。
    """
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
    df = pd.read_csv(file_path)
    
    # 3. 數據清洗與排序
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 4. 執行與特徵工程腳本 *完全相同* 的特徵創建步驟
    df['Price_vs_EMA20'] = (df['Close'] - df['EMA20']) / df['Close']
    df['MACD_Line_Diff'] = df['EMA12'] - df['EMA26']
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
        FEATURES = [f for f in FEATURES if f in df.columns]

    # 6. 處理缺失值
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    df_final = df[FEATURES + [TARGET]].copy()
    df_final.dropna(inplace=True)
    
    # 7. 【關鍵】準備 X 和 y (T -> T+1)
    # X: 使用當前特徵 (T)
    X = df_final[FEATURES].copy()
    # y: 預測 *未來* (T+1) 的報酬率
    y = df_final[TARGET].shift(-1)
    
    # 刪除最後一行來對齊
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    X_values = np.nan_to_num(X.values, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    
    return X_values, y.values

# --- 2. 定義參數搜索範圍 ---

# 隨機森林 (Random Forest) 的參數
param_dist_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0]
}

# XGBoost 的參數
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# LightGBM 的參數
param_dist_lgbm = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 40, 50],
    'max_depth': [5, 10, 15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# SVR (支持向量機) 的參數
param_dist_svr = {
    'C': [0.1, 1, 10, 50],
    'epsilon': [0.01, 0.05, 0.1, 0.2],
    'kernel': ['linear', 'rbf'] # 測試兩種核函數
}

# --- 3. 執行調參 ---

def run_tuning(X, y):
    """
    對所有模型執行隨機搜索。
    """
    
    # 1. 初始化時序交叉驗證
    # test_size=int(len(X) * 0.1) 確保測試集不會太小
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=int(len(X) * 0.1))
    
    # 2. 準備數據
    # SVR 對尺度敏感，樹模型不受影響。我們統一進行標準化。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 定義評分標準 (RMSE，越低越好)
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

    models_to_tune = {
        'rf': (RandomForestRegressor(random_state=42, n_jobs=-1), param_dist_rf),
        'xgb': (xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'), param_dist_xgb),
        'lgbm': (lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), param_dist_lgbm),
        'svr': (SVR(), param_dist_svr),
    }
    
    best_params_all = {}

    for name, (model, params) in models_to_tune.items():
        print(f"\n--- 正在調整 {name.upper()} ---")
        start_time = time()
        
        # 使用 SVR 時，我們傳入 X_scaled；樹模型其實不需要
        data_to_use = X_scaled if name == 'svr' else X
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=N_ITER,
            cv=tscv,
            scoring=rmse_scorer,
            n_jobs=-1, # 使用所有 CPU 核心
            random_state=42,
            verbose=1 # 顯示進度
        )
        
        search.fit(data_to_use, y)
        
        end_time = time()
        print(f"--- {name.upper()} 調整完成，耗時: {end_time - start_time:.2f} 秒 ---")
        print(f"最佳 RMSE 分數: {search.best_score_:.6f}")
        print(f"最佳參數: {search.best_params_}")
        
        best_params_all[name] = search.best_params_
        
    return best_params_all

# --- 4. 主執行緒 ---
if __name__ == "__main__":
    try:
        print("--- 步驟 1: 載入並準備數據 ---")
        X_data, y_data = load_and_prepare_data_for_tuning(DATA_FILE, FEATURES_FILE)
        
        print(f"\n數據準備完成。 X 形狀: {X_data.shape}, y 形狀: {y_data.shape}")
        
        print("\n--- 步驟 2: 開始超參數調整 (這可能需要很長時間) ---")
        print(f"交叉驗證折數 (Splits): {N_SPLITS}")
        print(f"隨機搜索次數 (Iterations per model): {N_ITER}")
        
        final_best_params = run_tuning(X_data, y_data)
        
        print("\n--- 步驟 3: 保存最佳參數 ---")
        with open(HYPERPARAMS_OUTPUT_FILE, 'w') as f:
            json.dump(final_best_params, f, indent=4)
            
        print(f"✅ 成功將所有模型的最佳參數保存到: {HYPERPARAMS_OUTPUT_FILE}")
        
    except FileNotFoundError as e:
        print(f"\n[執行失敗] 錯誤: {e}")
        print(f"請確保 '{DATA_FILE}' 和 '{FEATURES_FILE}' 都在同一個目錄下。")
    except Exception as e:
        print(f"\n[執行失敗] 發生未預期的錯誤: {e}")

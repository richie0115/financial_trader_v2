import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression 
import warnings

warnings.filterwarnings('ignore')

#  參數設定
FILE_NAME = 'AAPL_historical_data.csv' 
TARGET = 'daily_return' 
DATE_COLUMN = 'Date'
FINAL_FEATURES_OUTPUT_FILE = 'final_features.json' # 新增：導出特徵列表的檔案名稱

# 原始特徵
ORIGINAL_FEATURES = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume", 
    "RSI14", "RSI28", "RSI50", "EMA20", "EMA50", "EMA90", "EMA12", "EMA26", 
    "MACD", "AROON_bull_25", "AROON_bear_25", "AROON_bull_50", "AROON_bear_50", 
    "VROC_20", "DJI_Close", "WMA_14", "WMA_20", "WMA_200", "STD_20", "SMA_20", 
    "UB", "LB", "RVI_10"
]

# 數據載入改索引
print(f"數據載 {FILE_NAME}")

try:
    df = pd.read_csv(FILE_NAME, index_col=DATE_COLUMN, parse_dates=True)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            print("索引已成功轉換為日期時間格式 (DatetimeIndex)。")
        except Exception as e:
            print(f"data不給轉")
            exit()
            
    required_cols = ORIGINAL_FEATURES + [TARGET]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"錯誤：CSV 檔案中缺少必要的欄位: {missing_cols}")
        exit()
        
    df = df[required_cols].copy()
    
    # 關鍵修復：確保數據是按時間順序排列的
    df.sort_index(inplace=True)
    
    print(f"成功載入數據集，大小: {df.shape}")
    
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {FILE_NAME}。請檢查檔案路徑和名稱。")
    exit()

# 特徵工程：創建新的關係和時間序列特徵
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """創建新的衍生特徵"""
    df = data.copy()

    # 捕捉相對動量和位置特徵
    df['Price_vs_EMA20'] = (df['Close'] - df['EMA20']) / df['Close']
    df['MACD_Line_Diff'] = df['EMA12'] - df['EMA26']
    df['RSI14_Diff'] = df['RSI14'].diff()
    df['Aroon_Oscillator_25'] = df['AROON_bull_25'] - df['AROON_bear_25']
    df['BB_Percent_B'] = (df['Close'] - df['LB']) / (df['UB'] - df['LB'] + 1e-6)

    # 波動性強化特徵
    df['Volatility_Ratio'] = df['STD_20'] / (df['SMA_20'] + 1e-6) # 增加 1e-6 避免除以零
    
    # 高低價差（使用 shift(1) 避免當日資訊洩漏）
    # High_Low_Ratio 應該基於前一天的收盤價，以標準化當天的波動
    df['High_Low_Range'] = (df['High'] - df['Low']) / (df['Close'].shift(1) + 1e-6)

    # 納入滯後資訊 (Lagged Features)
    df['Lagged_DJI_Return'] = df['DJI_Close'].pct_change().shift(1)
    df['Lagged_Volume'] = df['Volume'].shift(1)
    df['Lagged_Return_1d'] = df[TARGET].shift(1) # 增加 1 天前的自身報酬率/因握用前一天

    # 分類特徵
    df['DayOfWeek'] = df.index.dayofweek
    df['RSI14_State'] = pd.cut(df['RSI14'], bins=[0, 30, 70, 100], 
                                labels=['Oversold', 'Neutral', 'Overbought'], right=False)

    # 讀熱
    df = pd.get_dummies(df, columns=['RSI14_State', 'DayOfWeek'], drop_first=True)

    # 由於創建了滯後和 diff 特徵，會產生 NaN，在這裡刪除它們
    return df.dropna().copy()

df_fe = feature_engineering(df)
NEW_FEATURES = [col for col in df_fe.columns if col not in ORIGINAL_FEATURES + [TARGET]]
ALL_FEATURES = ORIGINAL_FEATURES + NEW_FEATURES

print(f"\n[特徵工程完成] 新增特徵數: {len(NEW_FEATURES)}")
print(f"總特徵數: {len(ALL_FEATURES)}")

# 定義特徵和目標
X = df_fe[ALL_FEATURES]
y = df_fe[TARGET]

# --- 數據預處理：標準化 ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


# 斯皮爾曼相關性與非線性 MI Score
print("\n斯皮爾曼相關性與 Mutual Information (MI) Score")

spearman_corr = X_scaled.corrwith(y, method='spearman').abs().sort_values(ascending=False)
print("與 Target (daily_return) 斯皮爾曼相關性前")
print(spearman_corr.head(10))

mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Mutual Information Score 前 10 名 (衡量非線性關係):")
print(mi_scores.head(10))

MI_THRESHOLD = 0.005
low_relevance_features = mi_scores[mi_scores < MI_THRESHOLD].index.tolist()
X_mi = X_scaled.drop(columns=low_relevance_features)
print(f"\n根據 MI Score (< {MI_THRESHOLD}) 剔除特徵數: {len(low_relevance_features)}")
print(f"剩餘特徵數: {X_mi.shape[1]}")


# 共線性處理：檢查特徵之間的相關性矩陣
print("\共線性處理")

corr_matrix = X_mi.corr().abs()
UPPER_TRI = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

CORR_THRESHOLD = 0.95

MI_DIFF_THRESHOLD = 0.05 

high_corr_features = [] # 重新使用 list 或 set

for column in UPPER_TRI.columns:
    # 找到所有與 'column' 高度相關的特徵
    collinear_with = UPPER_TRI.index[UPPER_TRI[column] > CORR_THRESHOLD].tolist()
    
    if collinear_with:
        col_mi = mi_scores.get(column, 0)
        
        # 遍歷所有與 'column' 高度相關的特徵
        for related_feature in collinear_with:
            related_mi = mi_scores.get(related_feature, 0)
            
            # --- 調整後的比較邏輯 ---
            
            # 1. 計算 MI 的相對差異 (假設 MI > 0, 如果 MI=0 則需特殊處理)
            
            # 判斷是否應該移除 column： column 的 MI 顯著低於 related_feature
            # (col_mi * (1 + MI_DIFF_THRESHOLD) < related_mi)
            # 例如：如果 col_mi * 1.05 < related_mi，表示 related_mi 至少比 col_mi 高 5%
            
            # 為了避免除以零或 MI 接近零時的極端情況，使用絕對差異 + 門檻可能更安全。
            # 簡單方式：使用相對差異判斷
            
            is_col_significantly_lower = False
            is_related_significantly_lower = False

            if related_mi > 0 and col_mi > 0:
                # 判斷 column 是否顯著較低 (related_mi/col_mi > 1 + MI_DIFF_THRESHOLD)
                if related_mi / col_mi > (1 + MI_DIFF_THRESHOLD):
                    is_col_significantly_lower = True
                
                # 判斷 related_feature 是否顯著較低 (col_mi/related_mi > 1 + MI_DIFF_THRESHOLD)
                elif col_mi / related_mi > (1 + MI_DIFF_THRESHOLD):
                    is_related_significantly_lower = True
            
            elif col_mi == 0 and related_mi > 0:
                # 如果 column 的 MI 是 0，且 related_feature 的 MI 大於 0，則視為 column 顯著較低
                is_col_significantly_lower = True

            # --- 執行移除判斷 ---

            if is_col_significantly_lower:
                # 如果 'column' 的 MI 顯著較低，則將其加入移除列表
                if column not in high_corr_features:
                    print(f"高度共線性: {column} (MI: {col_mi:.4f}) 顯著低於 {related_feature} (MI: {related_mi:.4f}) - 移除 {column}")
                    high_corr_features.append(column)
                break # 已經決定移除 'column'，跳到下一個 'column'
            
            elif is_related_significantly_lower:
                # 如果 'related_feature' 的 MI 顯著較低，則將其加入移除列表
                if related_feature not in high_corr_features:
                    print(f"高度共線性: {related_feature} (MI: {related_mi:.4f}) 顯著低於 {column} (MI: {col_mi:.4f}) - 移除 {related_feature}")
                    high_corr_features.append(related_feature)

final_features_list_pre_model = [f for f in X_mi.columns if f not in high_corr_features]
X_pre_model = X_mi[final_features_list_pre_model]

print(f"共線性處理後剩餘特徵數: {X_pre_model.shape[1]}")


#  模型驗證：XGBoost 與置換重要性 (Permutation Importance)
print("\nXGBoost 與置換")

# 關鍵：股票預測必須使用時序分割，不能隨機打亂
TRAIN_RATIO = 0.8
split_point = int(len(X_pre_model) * TRAIN_RATIO)
X_train, X_test = X_pre_model.iloc[:split_point], X_pre_model.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

#  XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

#手動計算 RMSE
y_pred = xgb_model.predict(X_test)
base_mse = mean_squared_error(y_test, y_pred)
base_rmse = np.sqrt(base_mse) 
print(f"模型基準 RMSE (測試集): {base_rmse:.6f}")

# 修改評分函數，返回 RMSE
def score_func(model, X, y):
    return np.sqrt(mean_squared_error(y, model.predict(X)))

# 計算置換重要性
feature_importances = {}
for col in X_pre_model.columns:
    X_test_permuted = X_test.copy()
    X_test_permuted[col] = np.random.permutation(X_test_permuted[col].values)
    
    permuted_rmse = score_func(xgb_model, X_test_permuted, y_test)
    
    # 重要性 = 錯誤增加的量（越高越重要）
    feature_importances[col] = permuted_rmse - base_rmse

importance_series = pd.Series(feature_importances).sort_values(ascending=False)

print("/n特徵置換重要性排名")
print(importance_series.head(15))

N_FEATURES_TO_KEEP = 17 # 您想要保留的特徵數量

# 最終決策
print("\n  最終決策")

# 選擇置換重要性排名前 N 的特徵
final_selection = importance_series.head(N_FEATURES_TO_KEEP).index.tolist()

print(f"\n[最終選定特徵集] 數量: {len(final_selection)} (強制保留前 {N_FEATURES_TO_KEEP} 名)")
print("最終特徵列表:")
print(final_selection)
# 最終決策
print("\n  最終決策")

final_selection = importance_series[importance_series > 0].index.tolist()

if not final_selection:
    print("警告：沒有特徵的置換重要性大於 0。已改為選擇排名前 10 的特徵作為備選。")
    final_selection = importance_series.head(20).index.tolist()#調特徵量

print(f"\n[最終選定特徵集] 數量: {len(final_selection)}")
print("最終特徵列表:")
print(final_selection)

# 最終訓練的數據集 (僅供參考)
X_final_model = X_pre_model[final_selection]
print(f"\n最終訓練數據 (X_final_model) 形狀: {X_final_model.shape}")



print(f"\n--- 6. 輸出最終分析結果到 {FINAL_FEATURES_OUTPUT_FILE} ---")

# 將最終的特徵列表儲存為 JSON，供模型預測腳本使用
try:
    with open(FINAL_FEATURES_OUTPUT_FILE, 'w') as f:
        json.dump(final_selection, f)
    print(f"最終特徵列表已成功保存到文件: {FINAL_FEATURES_OUTPUT_FILE}")
except Exception as e:
    print(f"錯誤：保存特徵列表到 JSON 文件時失敗: {e}")

# (可選) 儲存詳細的重要性報告 CSV
OUTPUT_FILENAME_CSV = 'AAPL_Feature_Selection_Results.csv'
try:
    # 創建一個包含最終選定特徵及其重要性的 DataFrame
    final_importance_df = importance_series.loc[final_selection].to_frame(name='Permutation_Importance')

    # 增加一欄，標註該特徵是原始特徵還是新創建的特徵
    def get_feature_source(feature_name):
        if feature_name in ORIGINAL_FEATURES:
            return 'Original'
        elif feature_name in NEW_FEATURES:
            return 'Engineered'
        # 處理 One-Hot Encoding 後的分類特徵
        elif any(feature_name.startswith(f) for f in ['RSI14_State_', 'DayOfWeek_']):
            return 'Engineered_Categorical'
        else:
            return 'Unknown'

    final_importance_df['Feature_Source'] = final_importance_df.index.map(get_feature_source)
    final_importance_df = final_importance_df.sort_values(by='Permutation_Importance', ascending=False)
    final_importance_df.to_csv(OUTPUT_FILENAME_CSV)
    print(f"✅ 詳細的重要性分析報告已保存到文件: {OUTPUT_FILENAME_CSV}")
except Exception as e:
    print(f"錯誤：保存重要性 CSV 報告時失敗: {e}")
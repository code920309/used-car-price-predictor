import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

def clean_numeric(val):
    if pd.isna(val):
        return 0
    # 문자와 특수문자 제거 후 숫자로 추출
    cleaned = ''.join(c for c in str(val) if c.isdigit())
    return int(cleaned) if cleaned else 0

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def train_model():
    # 1. 데이터 로드
    file_name = 'data/used_cars_dataset_v2.csv'
    
    try:
        df = pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, encoding='cp949')
    
    # 2. 전처리 및 정제
    if 'AskPrice' in df.columns:
        df['AskPrice'] = df['AskPrice'].apply(clean_numeric)
    if 'kmDriven' in df.columns:
        df['kmDriven'] = df['kmDriven'].apply(clean_numeric)
    
    # 학습에 사용할 특성 선택
    # Brand, Year, kmDriven, Transmission, FuelType, Owner 등을 학습에 활용
    features = ['Brand', 'Year', 'kmDriven', 'Transmission', 'FuelType', 'Owner']
    target = 'AskPrice'
    
    df = df.dropna(subset=features + [target])
    
    X = df[features].copy()
    y = df[target]
    
    encoders = {}
    categorical_cols = ['Brand', 'Transmission', 'FuelType', 'Owner']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # 3. 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 모델 학습 및 평가
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2, 'model': model}
        
    # 결과 출력 및 비교표 작성
    print("\n[전체 모델 성능 비교 결과]")
    print(f"{'Model Name':<20} | {'RMSE':<15} | {'R2 Score':<10}")
    print("-" * 50)
    for name, metric in results.items():
        print(f"{name:<20} | {metric['RMSE']:<15.2f} | {metric['R2']:<10.4f}")

    # 5. 최적의 모델 선정 (R2 스코어 기준)
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n최고 성능 모델: {best_model_name} (R2: {results[best_model_name]['R2']:.4f})")

    # 6. 모델 및 전처리 규칙 저장
    if not os.path.exists('models'):
        os.makedirs('models')
        
    save_data = {
        'model': best_model,
        'encoders': encoders,
        'features': features
    }
    
    with open('models/car_model.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"모델 저장 완료: models/car_model.pkl ({best_model_name} 로드됨)")

if __name__ == "__main__":
    train_model()

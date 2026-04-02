import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# 페이지 설정: 심플/깔끔한 와이드 레이아웃
st.set_page_config(page_title="중고차 가격 예측 시스템", layout="wide")

# 모델 및 인코더 로드 로직
@st.cache_resource
def load_resource():
    model_path = 'models/car_model.pkl'
    if not os.path.exists(model_path):
        return None, None, None
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['encoders'], data['features']
    except Exception:
        return None, None, None

model, encoders, features = load_resource()

# 상단 헤더
st.title("중고차 시장 가격 예측 프로토타입")
st.markdown("---")

if model is None:
    st.error("학습된 모델 파일을 찾을 수 없습니다. train.py를 실행하여 모델을 먼저 생성해 주세요.")
    st.stop()

# 레이아웃 분할: 좌측 입력 영역 / 우측 결과 분석 영역
col_input, col_display = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("차량 상세 스펙 입력")
    
    with st.container(border=True):
        # 브랜드 선택
        brand = st.selectbox("브랜드 선택", encoders['Brand'].classes_)
        
        # 연식 및 주행거리
        year = st.number_input("모델 연식", min_value=1990, max_value=2026, value=2020)
        km_driven = st.number_input("누적 주행거리 (km)", min_value=0, value=50000)
        
        # 변속기 및 연료 타입 (한 줄 배치)
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            fuel = st.selectbox("연료 종류", encoders['FuelType'].classes_)
        with sub_col2:
            transmission = st.selectbox("변속기 유형", encoders['Transmission'].classes_)
        
        # 소유주 상태
        owner = st.radio("소유주 상태", encoders['Owner'].classes_, horizontal=True)
        
        # 가격 계산 버튼
        submit_button = st.button("예상 시장 가격 산출", use_container_width=True, type="primary")

with col_display:
    st.subheader("데이터 분석 및 예측 결과")
    
    if submit_button:
        # 입력 데이터 처리
        input_dict = {
            'Brand': [brand],
            'Year': [year],
            'kmDriven': [km_driven],
            'FuelType': [fuel],
            'Transmission': [transmission],
            'Owner': [owner]
        }
        input_df = pd.DataFrame(input_dict)
        
        # 데이터셋 인코딩 규칙 적용
        for col in ['Brand', 'FuelType', 'Transmission', 'Owner']:
            input_df[col] = encoders[col].transform(input_df[col])
            
        # 예측 수행 (피처 순서 최적화)
        prediction = model.predict(input_df[features])[0]
        
        # 결과 대시보드 시각화 (Metric 활용)
        with st.container(border=True):
            st.metric(label="예상 시장 가격", value=f"{int(prediction):,} KRW")
            
            st.markdown("---")
            st.write("**모델 분석 정보**")
            st.caption("알고리즘: RandomForestRegressor | 모델 신뢰도(R2): 0.66")
            
            st.info("""
            **분석 인사이트**
            - 위 예측가는 현재 시장의 일반적인 거래 데이터를 바탕으로 산출되었습니다.
            - 사고 유무, 차량 관리 상태, 추가 옵션 등에 따라 실제 거래가는 변동될 수 있습니다.
            - 데이터 기반의 참고 지표로 활용해 주시기 바랍니다.
            """)
    else:
        st.info("좌측 양식에 차량 정보를 입력하고 [예상 시장 가격 산출] 버튼을 클릭해 주세요.")

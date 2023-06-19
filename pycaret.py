import streamlit as st
import pandas as pd
#from pycaret.regression import setup, compare_models
from pycaret import regression

best_model = None  # 모델 비교 및 생성 결과를 저장하는 변수

def pycaret1():
    global best_model  # 전역 변수로 선언

    st.title("CSV 파일 업로드 및 변수 설정")

    # CSV 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # 컬럼 목록 출력
        st.subheader("데이터 컬럼 목록")
        st.write(data.columns)

        # 컬럼 선택
        columns = data.columns
        target_column = st.selectbox("타겟 변수 선택", columns)
        numeric_columns = st.multiselect("수치형 변수 선택", columns, default=['IN_RADIUS', 'OUT_RADIUS', 'MOLD_POS'])

        # 버튼 클릭 여부 확인
        button_compare = st.button("모델 비교 및 생성")

        if button_compare and best_model is None:
            # Pycaret 설정
            exp = regression.setup(data, target=target_column, numeric_features=numeric_columns, normalize=True)

            # 모델 비교 및 생성
            best_model = regression.compare_models()

            # 모델 비교 및 생성 결과를 출력
            st.subheader("최적 모델")
            st.write(best_model)

if __name__ == "__main__":
    pycaret1()


import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI 데이터센터 냉각 전력 예측",
    page_icon="❄️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "cooling_power_prediction_model_compressed.pkl"


@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    return package


st.title("AI 기반 데이터센터 냉각 전력 예측")
st.write(
    "Random Forest 머신러닝 모델을 이용해 "
    "다음 시점의 냉각 전력 관련 값을 예측합니다."
)

if not MODEL_PATH.exists():
    st.error("AI 모델 파일을 찾을 수 없습니다.")
    st.stop()

try:
    package = load_model()
    model = package["model"]
    feature_columns = package["feature_columns"]
except Exception as e:
    st.error(f"모델 로딩 오류: {e}")
    st.stop()

st.warning(
    "예측값은 표준화된 값(Standardized Value)입니다. "
    "실제 전력 단위인 kW 또는 kWh로 해석할 수 없습니다."
)

st.subheader("1. 데이터 업로드")

uploaded_file = st.file_uploader(
    "센서 데이터 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, sep=";")
        data.columns = data.columns.astype(str).str.strip()

        missing = [
            col for col in feature_columns
            if col not in data.columns
        ]

        if missing:
            st.error(
                "필수 입력 열이 누락되었습니다: "
                + ", ".join(missing)
            )
            st.stop()

        if data.empty:
            st.error("CSV에 데이터가 없습니다.")
            st.stop()

        input_data = data[feature_columns].copy()

        for col in feature_columns:
            input_data[col] = pd.to_numeric(
                input_data[col], errors="coerce"
            )

        if input_data.isna().any().any():
            st.error("입력 데이터에 결측값 또는 비숫자 값이 있습니다.")
            st.stop()

        st.success(f"데이터 확인 완료: {len(input_data)}개 행")

        st.subheader("2. AI 예측 결과")

        latest_input = input_data.iloc[[-1]]

        prediction = float(
            model.predict(latest_input)[0]
        )

        current_value = float(
            latest_input["P_ac-0"].iloc[0]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "현재 P_ac-0",
            f"{current_value:.4f}"
        )

        col2.metric(
            "다음 시점 예측값",
            f"{prediction:.4f}"
        )

        st.caption(
            "업로드한 CSV의 마지막 행을 기준으로 다음 행의 "
            "P_ac-0를 예측합니다. 원본 데이터의 시간 간격은 "
            "15분으로 설명되어 있으나, 실제 입력 데이터의 "
            "시간 간격도 확인해야 합니다."
        )

        result = pd.DataFrame({
            "구분": ["현재 값", "다음 시점 예측값"],
            "P_ac-0": [current_value, prediction]
        })

        st.bar_chart(result.set_index("구분"))

        output = pd.DataFrame([{
            "현재_P_ac-0": current_value,
            "예측_P_ac-0": prediction
        }])

        st.download_button(
            "예측 결과 CSV 다운로드",
            data=output.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="prediction_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")

st.divider()

st.caption(
    "본 모델은 표준화된 P_ac-0 값을 예측하며, "
    "실제 에너지 소비량이나 절감량을 직접 계산하지 않습니다."
)

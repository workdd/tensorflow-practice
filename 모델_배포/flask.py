import json
from flask import Flask, request
from tensorflow.keras.models import load_model

# 데이터 구조를 변환하기 위한 전처리 진행
from utils import preprocess

# 훈련된 모델 로드
model = load_model('model.h5')
app = Flask(__name__)


@app.route('classify', methods=['POST'])
def classify():
    complaint_data = request.form["complaint_data"]
    preprocessed_complaint_data = preprocess(complaint_data)
    # 예측 수행
    prediction = model.predict([preprocessed_complaint_data])
    # HTTP 응답에서 예측을 반환
    result = json.dumps({"score": prediction})
    return result

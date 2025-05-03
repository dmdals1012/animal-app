🐶 프로젝트 개요
강아지 표정 감정 분류 AI 웹앱
Teachable Machine으로 학습시킨 강아지 감정 분류 모델을 Streamlit을 통해 웹 서비스로 배포합니다. 사용자가 강아지 이미지를 업로드하면 행복/슬픔/화남/평온 등 4가지 감정 상태를 분석해 결과를 시각적으로 제공합니다.

Streamlit app : https://animal-app-nvrvwmmvsos8wyiuaowkkh.streamlit.app/

🚀 주요 기능
    - 이미지 업로드 인터페이스: JPG/PNG/JPEG/WEBP 형식 지원

    - 실시간 감정 분석: 업로드 즉시 예측 결과 표시

    - 시각화 시스템

        - 최종 감정 상태 텍스트 표시

        - 예측 신뢰도 % 수치화

        - 모든 클래스별 확률 분포 막대그래프 제공

⚙️ 기술 스택

```bash
**AI 모델**: Teachable Machine (TensorFlow/Keras)
**웹 프레임워크**: Streamlit
**이미지 처리**: PIL, NumPy
**배포**: Streamlit Cloud
```

🐾 Teachable Machine 모델 생성 과정

- 사이트 접속: Teachable Machine

- 프로젝트 유형: 이미지 프로젝트 → 표준 이미지 모델 선택

- 클래스 구성:
```bash
- Class 1: 행복 (약 1000장)
- Class 2: 슬픔 (약 1000장) 
- Class 3: 화남 (약 1000장)
- Class 4: 평온 (약 1000장)
```


2. 데이터 준비
- 데이터셋: Kaggle Dog Emotion Dataset 사용

- 분할 비율:
    학습용: 80% 
    검증용: 20%

3. 학습 설정
```bash
- **에포크(Epochs)**: 50 
- **배치 크기**: 16
- **데이터 증강**: 
  - 좌우 반전
  - 회전(±10도)
  - 확대/축소
  - 밝기 조절
```

4. 모델 학습 실행

- 학습시키기 버튼 클릭 → 약 10~15분 소요

- 목표 성능: 검증 정확도 ≥70%


5. 모델 테스트

- 이미지 업로드 테스트: 10장 이상 검증


6. 모델 내보내기
```bash
    1. `모델 내보내기` → `TensorFlow` → `Keras` 선택
    2. 다운로드 파일:
    - keras_model.h5 (모델 파일)
    - labels.txt (레이블 정보)
    3. 프로젝트 구조:
    ├─ model/
    │   ├─ keras_model.h5
    │   └─ labels.txt
```


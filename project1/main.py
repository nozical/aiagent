import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. 데이터 로드
# 현재 작업 디렉토리의 data 폴더에 있는 파일을 사용합니다.
file_path = os.path.join('data', '2_PAproject_2_4_machine.csv')

try:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"데이터를 성공적으로 불러왔습니다: {file_path}")
    else:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
except FileNotFoundError as e:
    print(e)
    # 파일이 없는 경우를 대비해 스크립트 중단 또는 예외 처리
    exit(1)

# 2. 독립변수(X)와 종속변수(y) 설정
# 원인변수: 부서, 성과등급, 급여, 근무시간
# 결과변수: 퇴직 여부(Left)
X = df[['Department', 'Performance_Rating', 'Salary', 'Work_Hours']]
y = df['Left']

# 3. 전처리 파이프라인 설정
# - Department: 범주형 변수이므로 One-Hot Encoding 적용
# - 나머지: 수치형 변수이므로 SVM의 성능 향상을 위해 Standard Scaling 적용
categorical_features = ['Department']
numeric_features = ['Performance_Rating', 'Salary', 'Work_Hours']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. SVM 모델 구성 (확률 예측을 위해 probability=True 설정)
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
])

# 5. 모델 학습
# 실제 프로젝트에서는 train_test_split으로 검증을 수행하는 것이 좋습니다.
svm_model.fit(X, y)
print("모델 학습이 완료되었습니다.")

# 6. 배치 예측 (2_PAproject_2_4_machine_prediction.csv 활용)
prediction_file_path = os.path.join('data', '2_PAproject_2_4_machine_prediction.csv')

try:
    if os.path.exists(prediction_file_path):
        predict_df = pd.read_csv(prediction_file_path)
        print(f"\n예측용 데이터를 불러왔습니다: {prediction_file_path}")
        
        # 모델을 사용하여 예측 수행
        predictions = svm_model.predict(predict_df)
        pred_probas = svm_model.predict_proba(predict_df)
        
        # 결과 데이터프레임 생성
        result_df = predict_df.copy()
        result_df['Prediction'] = ['이직(Left)' if p == 1 else '잔류(Stay)' for p in predictions]
        result_df['Attrition_Probability (%)'] = np.round(pred_probas[:, 1] * 100, 2)
        
        # 7. 결과 출력
        print("\n=== 배치 예측 결과 ===")
        print(result_df)
        
        # 결과를 CSV 파일로 저장 (선택 사항)
        output_path = os.path.join('data', '2_PAproject_2_4_machine_results.csv')
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n예측 결과가 저장되었습니다: {output_path}")
        
    else:
        print(f"예측할 파일({prediction_file_path})이 없습니다.")
except Exception as e:
    print(f"예측 중 오류 발생: {e}")


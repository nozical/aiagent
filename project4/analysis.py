import os
import sys
from io import StringIO

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# 분석 로그를 파일과 콘솔에 동시에 출력하기 위한 설정
log_buffer = StringIO()

class Tee:
    """콘솔과 파일에 동시에 출력"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.__stdout__, log_buffer)

# 1. 데이터 불러오기
file_path = os.path.join(DATA_DIR, "5_PAproject_5_4_rater.xlsx")
df = pd.read_excel(file_path)

# 범주형 변수 변환 (Patsy 호환성 확보)
categorical_cols = ['department', 'job_level', 'rater_id']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# 결측치 제거
df = df.dropna(subset=['rating_score', 'performance_true', 'goal_difficulty'])

print("--- [1] 데이터 로드 및 타입 변환 완료 ---")
print(df.dtypes)

# 2. 기술통계
rater_stats = df.groupby('rater_id', observed=True)['rating_score'].agg(['count', 'mean', 'std']).reset_index()
print("\n--- [2] 평가자별 기술통계 ---")
print(rater_stats.to_string())

# 3. ANOVA 분석
groups = [group['rating_score'].values for name, group in df.groupby('rater_id', observed=True)]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\n--- [3] ANOVA 결과: F={f_stat:.4f}, p-value={p_val:.4f} ---")

# 4. HLM (혼합 효과 모형) 적합
model_formula = "rating_score ~ performance_true + goal_difficulty + age + tenure_years + department + job_level"
hlm_model = smf.mixedlm(model_formula, df, groups=df["rater_id"])
hlm_result = hlm_model.fit()

print("\n--- [4] HLM 분석 요약 ---")
print(hlm_result.summary())

# 5. ICC 계산
var_resid = hlm_result.scale
var_rater = float(hlm_result.cov_re.iloc[0, 0])
icc = var_rater / (var_rater + var_resid)
print(f"\n--- [5] ICC: {icc:.4f} ---")

# 6. 평가자별 Random Effect (Bias) 추출
random_effects = hlm_result.random_effects
bias_df = pd.DataFrame.from_dict(random_effects, orient='index').reset_index()
bias_df.columns = ['rater_id', 'random_effect']

def judge_bias(x):
    if x > 0.1:   return 'Leniency (관대화)'
    elif x < -0.1: return 'Severity (엄격화)'
    else:          return 'Neutral (중립)'

bias_df['bias_type'] = bias_df['random_effect'].apply(judge_bias)
print("\n--- [6] 평가자별 편향 분석 ---")
print(bias_df.sort_values(by='random_effect', ascending=False).to_string())

# 7. 평가점수 보정값 계산
df = df.merge(bias_df, on='rater_id', how='left')
df['adjusted_rating_score'] = df['rating_score'] - df['random_effect']

print("\n--- [7] 보정된 평가 점수 샘플 ---")
print(df[['employee_id', 'rater_id', 'rating_score', 'random_effect', 'adjusted_rating_score']].head().to_string())

# 결과 저장
excel_path = os.path.join(RESULT_DIR, "adjusted_results.xlsx")
df.to_excel(excel_path, index=False)
print(f"\n결과 엑셀 저장 완료: {excel_path}")

# 분석 로그 텍스트 파일 저장
sys.stdout = sys.__stdout__  # stdout 복구
log_text = log_buffer.getvalue()
log_path = os.path.join(RESULT_DIR, "analysis_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(log_text)
print(f"분석 로그 저장 완료: {log_path}")
print("모든 분석 완료!")

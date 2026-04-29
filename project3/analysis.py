import os
import matplotlib
matplotlib.use('Agg')  # 화면 없는 서버 환경용 백엔드 설정

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib  # 한글 폰트 설정 (koreanize-matplotlib)

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# 1. 데이터 로드
file_path = os.path.join(DATA_DIR, "2_PAproject_2_3_EDA.csv")
df = pd.read_csv(file_path)
print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")

# 이탈 여부 파생 변수 생성 (Voluntary, Involuntary → 1, 재직 중 → 0)
df['Is_Terminated'] = df['Status'].apply(lambda x: 1 if x in ['Voluntary', 'Involuntary'] else 0)

# ---------------------------------------------------------
# 2. 부서(Department)와 직무(Job_Role)별 이탈률 히트맵
# ---------------------------------------------------------
heatmap_data = df.pivot_table(
    index='Department', columns='Job_Role',
    values='Is_Terminated', aggfunc='mean'
) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd',
            linewidths=.5, cbar_kws={'label': '이탈률 (%)'})
plt.title('부서 및 직무별 이탈률(Attrition Rate) 히트맵 (%)', fontsize=16, pad=20)
plt.xlabel('직무 (Job Role)', fontsize=12)
plt.ylabel('부서 (Department)', fontsize=12)
plt.tight_layout()

heatmap_path = os.path.join(RESULT_DIR, "1_heatmap_attrition.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"히트맵 저장 완료: {heatmap_path}")

# ---------------------------------------------------------
# 3. 성과등급(Performance_Rating)별 이탈률 막대그래프
# ---------------------------------------------------------
rating_attrition = df.groupby('Performance_Rating')['Is_Terminated'].mean() * 100

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=rating_attrition.index, y=rating_attrition.values, palette='viridis')

plt.title('성과등급(Performance Rating)별 이탈률 (%)', fontsize=16, pad=20)
plt.xlabel('성과등급 (Rating)', fontsize=12)
plt.ylabel('이탈률 (%)', fontsize=12)
plt.ylim(0, max(rating_attrition.values) * 1.2)

for i, v in enumerate(rating_attrition.values):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

barplot_path = os.path.join(RESULT_DIR, "2_barplot_performance.png")
plt.savefig(barplot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"막대그래프 저장 완료: {barplot_path}")

print("\n모든 분석 완료!")

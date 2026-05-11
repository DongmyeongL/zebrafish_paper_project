# Figure Evaluation Results

평가일: 2026-05-02

평가 기준:

- `evaluate_sicence_figure.md`의 자연 과학 논문 그림 평가 체크리스트를 기준으로 평가하였다.
- Figure legend, 본문 설명, methods 설명은 직접 반영하지 않고 그림 자체만 보고 평가하였다.
- 점수는 자연 과학 논문 제출용 figure 완성도를 기준으로 한 주관적 평가이다.

## 평가 대상

- `output/png/figure9_final.png`
- `output/png/figure12_final.png`
- `output/png/figure13_final.png`
- `output/png/figure_fc_dynamics_final_fc_shift_summary.png`
- `output/png/figure_sc_fc_final_overview.png`

## 점수 요약

| Figure | 점수 | 평가 |
|---|---:|---|
| `figure_fc_dynamics_final_fc_shift_summary.png` | 82/100 | 가장 깔끔한 편이다. 두 panel 모두 메시지가 명확하고 색상과 legend도 좋다. 다만 제목과 축 글자가 지나치게 커서 여백을 많이 차지하고, region label annotation 일부가 산점도와 가까워 보인다. |
| `figure13_final.png` | 74/100 | 모델 구조와 결과 흐름은 잘 보인다. 하지만 A-D panel 순서가 시각적으로 약간 혼란스럽고, 전체 제목, 축, legend가 과하게 커서 압박감이 있다. D panel의 x tick label도 기울어져 있고 여백이 부족하다. |
| `figure12_final.png` | 72/100 | 정보량과 구조가 좋고, heatmap, network, boxplot 흐름이 있다. 하지만 heatmap 오른쪽 row label과 colorbar 주변이 잘려 보이고, B의 network example들은 scale 또는 legend 없이 의미 해석이 어렵다. 하단 boxplot도 y-label 일부가 너무 크고 잘릴 위험이 있다. |
| `figure9_final.png` | 68/100 | 데이터와 분석은 풍부하지만 한 figure에 너무 많은 정보가 들어가 있다. A heatmap, B network, C-G boxplot이 모두 커서 전체적으로 밀도가 높고, B network는 label과 edge가 겹쳐 해석이 어렵다. 하단 boxplot 일부 y-label이 잘려 보이며 panel 간 균형도 조금 약하다. |
| `figure_sc_fc_final_overview.png` | 58/100 | 과학적 내용은 좋지만 시각적 완성도는 가장 손볼 곳이 많다. A heatmap의 숫자와 label이 너무 크고 서로 겹쳐 보이며, B scatter와 C-D-E-F bar plot 간 간격과 정렬이 불안정하다. x tick label이 길고 잘리거나 겹치며, 전체적으로 논문 figure보다는 초안 느낌이 강하다. |

## 수정 우선순위

1. `figure_sc_fc_final_overview`를 먼저 수정한다. 가장 큰 문제는 text overlap, 과도한 font size, panel 간 간격이다. 특히 A heatmap의 숫자 크기와 x/y label, B의 annotation 위치, E/F bar plot label을 줄이면 점수가 크게 오를 것으로 보인다.
2. `figure9`를 수정한다. B network panel을 줄이거나 supplementary figure로 분리하는 것을 고려할 만하다. 하단 C-G는 y-label 크기를 줄이고 significance bar 높이를 정리하면 더 안정된다.
3. `figure12`를 수정한다. heatmap 오른쪽 label/colorbar clipping을 해결하고, B network panel에 최소한의 설명 또는 scale/condition label을 명확히 넣으면 좋아진다.
4. `figure13`을 수정한다. 전체 font scale을 낮추고 panel 순서를 A-B-C-D로 자연스럽게 읽히게 재배치하면 좋아진다.
5. `figure_fc_dynamics_final_fc_shift_summary`는 minor polish만 필요하다. 제목과 축 글자를 조금 줄이고 annotation 위치를 다듬으면 제출용에 가깝다.

## 전체 평가

현재 그림들은 데이터와 분석 내용은 충분히 들어가 있지만, typography와 layout polish가 아직 덜 된 상태이다. 논문 제출용 기준으로는 `figure_fc_dynamics_final_fc_shift_summary`는 거의 가능하고, `figure13`과 `figure12`는 보정이 필요하며, `figure9`와 `figure_sc_fc_final_overview`는 구조적으로 다시 정리하는 것이 좋다.

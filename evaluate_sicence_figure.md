# 자연 과학 논문 그림 평가 체크리스트

자연 과학 논문 그림을 볼 때는 단순히 예쁜지보다, 독자가 데이터와 결론을 정확하게 이해할 수 있는지를 기준으로 평가한다. 아래 항목을 순서대로 확인한다.

## 1. 전체 메시지

- 이 그림이 전달하려는 핵심 질문이나 결론이 한눈에 보이는가?
- 각 panel이 서로 연결되어 하나의 논리적 흐름을 만드는가?
- 본문에서 주장하는 내용과 그림의 데이터가 일치하는가?
- 불필요한 장식이나 과도한 시각 효과가 데이터 해석을 방해하지 않는가?

## 2. 글자와 폰트

- 글자는 가능하면 Arial 또는 저널에서 요구하는 sans-serif 계열로 통일한다.
- x-tick, y-tick, x-label, y-label, panel label, colorbar tick, legend, title, 그래프 안 설명 글자의 크기가 체계적으로 정리되어 있는가?
- 같은 역할의 글자는 여러 panel에서 같은 크기와 스타일을 유지하는가?
- 글자가 너무 작아서 축소 인쇄 후 읽기 어렵지 않은가?
- 굵은 글씨, 기울임, 대문자 사용이 과하지 않고 일관적인가?

## 3. 축, 눈금, 단위

- x축과 y축 label이 명확한가?
- 모든 물리량, 생물학적 측정값, 시간, 거리, 농도 등에 단위가 표시되어 있는가?
- tick 개수가 너무 많거나 너무 적지 않은가?
- tick label이 서로 겹치지 않는가?
- 축 범위가 데이터를 왜곡하지 않는가?
- log scale, normalized value, z-score, fold change 등을 사용했다면 명확하게 표시했는가?
- 같은 변수를 비교하는 panel들은 가능한 한 같은 축 범위를 사용하는가?

## 4. Panel Figure 구성

- 각 panel의 크기와 비율이 적절하고, 비슷한 역할의 panel은 크기가 일정한가?
- panel 사이 간격이 일정하고 충분한 여유가 있는가?
- x축, y축, colorbar, legend, 제목의 정렬이 전체적으로 맞는가?
- panel이 너무 빽빽해서 독자가 순서를 따라가기 어렵지 않은가?
- panel 배치는 A, B, C 순서대로 자연스럽게 읽히는가?
- 대표 이미지와 정량 그래프가 함께 있을 경우, 서로 같은 조건과 순서를 따르는가?

## 5. Panel Label

- panel label은 A, B, C처럼 명확하게 표시되어 있는가?
- panel label 위치가 각 panel의 y-label 바로 왼쪽 또는 panel 좌상단에 일관되게 배치되어 있는가?
- panel label끼리 x축, y축 정렬이 맞는가?
- panel label이 데이터, 축, 이미지, legend와 겹치지 않는가?

## 6. Legend와 Annotation

- legend 글자가 겹치거나 너무 작지 않은가?
- legend가 데이터 영역을 가리지 않는가?
- 색, 선, marker, group 이름이 legend에서 명확하게 설명되는가?
- 그래프 안 annotation은 필요한 정보만 담고 있는가?
- 화살표, bracket, significance bar가 어떤 비교를 의미하는지 분명한가?

## 7. 색상과 접근성

- 색상은 그룹을 구분하기에 충분히 명확한가?
- colorblind-friendly palette를 사용했는가?
- 빨강-초록 조합처럼 색각 이상 독자에게 불리한 조합을 피했는가?
- heatmap이나 colorbar의 색 범위가 데이터 의미와 잘 맞는가?
- 같은 그룹이나 조건은 모든 panel에서 같은 색으로 유지되는가?
- 배경색, grid, 투명도 때문에 데이터가 흐려 보이지 않는가?

## 8. 데이터 표현의 정직성

- bar plot만 사용해서 분포를 숨기고 있지 않은가?
- 가능한 경우 개별 데이터 포인트, box plot, violin plot, swarm plot 등을 통해 분포를 보여주는가?
- error bar가 SEM, SD, CI 중 무엇인지 명확히 표시되어 있는가?
- sample size, subject 수, region 수, trial 수 등이 figure legend 또는 panel 안에 명시되어 있는가?
- outlier를 제거했다면 기준이 설명되어 있는가?
- 대표 이미지가 전체 데이터의 경향을 왜곡하지 않는가?

## 9. 통계 표시

- 어떤 통계 검정을 사용했는지 figure legend 또는 methods와 연결되어 있는가?
- p-value, corrected p-value, effect size, confidence interval 중 필요한 정보가 충분한가?
- multiple comparison correction이 필요한 상황에서 보정이 되었는가?
- 별표 표기만 있는 경우, 별표가 의미하는 p-value 기준이 설명되어 있는가?
- 통계적으로 유의하지 않은 결과도 해석상 중요하면 명확히 표시했는가?

## 10. 이미지와 해상도

- 현미경 이미지, 조직 이미지, 뇌 영역 이미지 등에는 scale bar가 있는가?
- scale bar의 길이와 단위가 명시되어 있는가?
- 이미지 contrast 조정이 과도하지 않은가?
- 이미지 crop이 비교 조건 간에 공정한가?
- 최종 출력 해상도가 저널 요구사항을 만족하는가?
- raster image는 흐릿하지 않고, vector 요소는 깨지지 않는가?

## 11. 일관성과 재현성

- figure 본문, legend, methods에서 사용하는 용어가 일치하는가?
- 약어는 처음 등장할 때 설명되어 있는가?
- region, condition, genotype, stimulus 등의 순서가 여러 panel에서 일관적인가?
- 데이터 처리 방식이 그림만 보고도 어느 정도 추적 가능한가?
- output figure와 함께 통계표 또는 source data가 제공되는가?

## 12. 최종 점검

- 축소해서 봐도 핵심 패턴과 글자가 보이는가?
- 흑백으로 인쇄해도 중요한 비교가 사라지지 않는가?
- 본문을 읽지 않아도 figure legend와 함께 그림을 이해할 수 있는가?
- 저널 형식에 맞는 파일 형식, 크기, 해상도, 폰트 조건을 만족하는가?
- 오탈자, 단위 누락, label 불일치, panel 순서 오류가 없는가?

## 13. title 
대부분 그림 title 필요없음 

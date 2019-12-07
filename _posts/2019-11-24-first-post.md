---
layout: post
title: python을 이용한 보스턴 범죄 분석(작성중)
gh-repo: 1cekrim/aix
gh-badge: [star, fork, follow]
tags: [aix]
comments: true
---

| <center>이름</center> | <center> 학과 </center> | <center>학번</center> | <center>이메일</center> |
| 김우찬 | 컴퓨터소프트웨어학부 | 2019031685 | lion0303@hanyang.ac.kr |
| 김현수 | 컴퓨터소프트웨어학부 | 2019027192 | 1cekrim.dev@gmail.com |
| 이경민 | 기계공학부           | 2019015196 | kminsmin@gmail.com |
| 조영환 | 원자력공학과         | 2014058386 | jyh4997@naver.com |
| 최민기 | 자연환경공학과       | 2017080746 | tjsrhr136@naver.com |

![crimes-in-boston-image](https://storage.googleapis.com/kaggle-datasets-images/49781/90388/0e523321547c24d989c910879491fce7/dataset-cover.JPG?t=2018-09-04-17-52-47)

## I. Introduction

### Motivation: Why are you doing this?
![Minority Report](https://ww.namu.la/s/06a79176babdc3957a67d9adbbc8614f934adea355ce3417d5596f9a236f50a6aa47b9aa4ad9d39ba60f03d8ed5a0dc419a86f9f915af409b6783259cc253aba33c4b3681cf5dc504a62c0f44f6af8131b7aa60c550d5afb0ad94f4addc633c0)<br>
2002년 개봉한 영화 ‘마이너리티 리포트’는 2054년 워싱턴을 배경으로 합니다.<br>
주 내용은 범죄가 일어나기 전 범죄를 예측해 범죄자를 단죄하는 최첨단 치안 시스템을 기반으로 하는데요? 이 시스템은 범죄가 일어날 시간과 장소, 범행을 저지를 사람들까지 예측할 수 있고, 특수경찰들이 미래의 범죄자들을 체포합니다.<br>
우리는 여기서 아이디어를 얻어 2019년 판 ‘피리크라임’을 만들기로 했습니다. 빅데이터와 딥러닝을 통해서 말이죠.<br>
우리는 2015년부터 2018년까지의 보스턴에서 발생한 범죄 데이터를 바탕으로. 언제 어디서 어떤 날씨에 어떤 범죄가 일어날지 예측하는 시스템을 만들고자 합니다.<br>
저의 목표는 2020년 보스턴에서의 첫 범죄를 예측하는 것입니다.

### What do you want to see at the end?
4년간의 보스턴의 범죄 데이터를 바탕으로, 앞으로 일어날 범죄를 예측하고 이를 바탕으로 범죄 예방에 초점을 두고 있습니다.<br>
범죄는 생각보다 불연속적 독립적 성격을 띠고 있지 않습니다. 오히려 연속적이면 비독립적이죠. 서로 연관이 되어 있습니다. 요일에 따라서 날씨에 따라서 범죄의 양뿐만 아니라 범죄의 종류까지 달라집니다.<br>
예를 들자면, 금요일에는 대출사기가 많은 비율을 차지했습니다. 금요일과 대출사기의 상관관계를 찾는다면, 토요일과 일요일은 은행 영업을 안 한다는 이유 때문이겠죠.<br>
궁극적으로는 범죄를 예측할 뿐 아니라 범죄와 여러 요건 간의 상관관계를 통해 확실한 범죄 예측 및 예방이 저희 팀 조의 최종 목표입니다. 

## II. Datasets

[캐글 주소](https://www.kaggle.com/AnalyzeBoston/crimes-in-boston)

2015년부터 2018년까지 보스턴에서 발생한 범죄를 통계낸 자료입니다.<br>
이 자료가 제공하는 csv 파일은 다음과 같습니다.<br>

| <center>파일 이름</center> | <center>설명</center> |
| crime.csv | 범죄가 일어난 시간, 요일, 범죄 코드 등이 나와있는 csv 파일 |
| offense_codes.csv | 범죄 코드와 범죄 이름이 연결되어 있는 csv 파일 |

crime.csv의 각 열은 다음과 같은 의미입니다.

| <center>열 이름</center> | <center>설명</center> |
| INCIDENT_NUMBER | 사건 번호. 모든 범죄는 각자 고유한 사건 번호를 하나씩 가집니다. |
| OFFENSE_CODE | 어떤 범죄를 저질렀는 지 알려주는 코드입니다. offense_codes.csv에서 찾을 수 있습니다. |
| OFFENSE_CODE_GROUP | 저지른 범죄가 어떤 종류에 속하는지 알려줍니다. |
| OFFENSE_DESCRIPTION | 범죄 묘사입니다. |
| DISTRICT | 어느 지구에서 범죄가 발생했는지 알려줍니다. |
| REPORTING_AREA | 어디서 보고되었는지 알려줍니다. |
| SHOOTING | 총격이 있었는지 알려줍니다. |
| OCCURRED_ON_DATE | 범죄 발생 날짜를 알려줍니다. |
| YEAR | 범죄가 발생한 년도입니다. |
| MONTH | 범죄가 발생한 달입니다. |
| DAY_OF_WEEK | 범죄가 발생한 요일입니다. |
| HOUR | 범죄가 발생한 시간입니다. |
| UCR_PART | 범죄 보고서의 어느 부분에 작성되어 있는지 알려줍니다. |
| STREET | 범죄가 발생한 도로입니다. |
| Lat | 범죄가 발생한 위치의 위도입니다. |
| Long | 범죄가 발생한 위치의 경도입니다. |
| Location | 범죄가 발생한 위치입니다. |

offense_codes.csv를 보면, 범죄 코드 하나에 범죄 이름이 여러 개가 연결되어 있는 등 문제가 좀 있습니다.<br>
그래서 보기좋게 전처리를 해주겠습니다.

```python
import pandas as pd
import re
import csv
# offense_codes.csv에 인코딩이 잘못된 문자가 있기 때문에, engine을 c에서 python으로 바꿔줘야 합니다.
df = pd.read_csv('offense_codes.csv', engine='python')
# CODE값을 기준으로 오름차순으로 정렬해줍니다.
df = df.sort_values(['CODE'])
# NAME 값에서 필요없는 문자들을 지워줍니다.
df['NAME'] = df['NAME'].map(lambda x: re.sub("[^a-zA-Z0-9-()$&.,/]", " ", x))
# 중복된 값들을 지줘줍니다.
df = df.drop_duplicates(subset='CODE', keep='first')
# fixed_offense_codes.csv에 결과물을 적어줍니다.
df.to_csv('fixed_offense_codes.csv', index=False)
```

## III. Methodology


## IV. Evaluation & Analysis


## V. Related Work
[Crime Type Classification using Neural Networks: A brief walkthrough](https://medium.com/@nicksypark/crime-type-classification-using-neural-networks-a-brief-walkthrough-841b273f9afe)(blog)

작성중입니다...
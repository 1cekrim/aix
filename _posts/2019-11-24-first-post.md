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

---
# I. Introduction

### Motivation: Why are you doing this?
![Minority Report](https://ww.namu.la/s/06a79176babdc3957a67d9adbbc8614f934adea355ce3417d5596f9a236f50a6aa47b9aa4ad9d39ba60f03d8ed5a0dc419a86f9f915af409b6783259cc253aba33c4b3681cf5dc504a62c0f44f6af8131b7aa60c550d5afb0ad94f4addc633c0)<br>
2002년 개봉한 영화 ‘마이너리티 리포트’는 2054년 워싱턴을 배경으로 합니다.<br>
주 내용은 범죄가 일어나기 전 범죄를 예측해 범죄자를 단죄하는 최첨단 치안 시스템을 기반으로 하는데요? 이 시스템은 범죄가 일어날 시간과 장소, 범행을 저지를 사람들까지 예측할 수 있고, 특수경찰들이 미래의 범죄자들을 체포합니다.<br>
우리는 여기서 아이디어를 얻어 2019년 판 ‘피리크라임’을 만들기로 했습니다. 빅데이터와 딥러닝을 통해서 말이죠.<br>
우리는 2015년부터 2018년까지의 보스턴에서 발생한 범죄 데이터를 바탕으로. 언제 어디서 어떤 날씨에 어떤 범죄가 일어날지 예측하는 시스템을 만들고자 합니다.<br>
저희의 목표는 2020년 보스턴에서의 첫 범죄를 예측하는 것입니다.

### What do you want to see at the end?
4년간의 보스턴의 범죄 데이터를 바탕으로, 앞으로 일어날 범죄를 예측하고 이를 바탕으로 범죄 예방에 초점을 두고 있습니다.<br>
범죄는 생각보다 불연속적 독립적 성격을 띠고 있지 않습니다. 오히려 연속적이면 비독립적이죠. 서로 연관이 되어 있습니다. 요일에 따라서 날씨에 따라서 범죄의 양뿐만 아니라 범죄의 종류까지 달라집니다.<br>
예를 들자면, 금요일에는 대출사기가 많은 비율을 차지했습니다. 금요일과 대출사기의 상관관계를 찾는다면, 토요일과 일요일은 은행 영업을 안 한다는 이유 때문이겠죠.<br>
궁극적으로는 범죄를 예측할 뿐 아니라 범죄와 여러 요건 간의 상관관계를 통해 확실한 범죄 예측 및 예방이 저희 팀 조의 최종 목표입니다. 

---

# II. Datasets

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

<br><br><br>


(데이터셋을 이용해 그린 그래프들이 들어갈 위치)


## 데이터 전처리 1

데이터셋을 보면, 불필요한 열이 있습니다.<br>
```INCIDENT_NUMBER```은 그냥 순서대로 부여되는 의미없는 번호일 뿐이니 필요 없습니다.<br>
```OFFENSE_CODE, OFFENSE_DESCRIPTION```은 ```OFFENSE_CODE_GROUP``` 만으로도 표현 가능하므로 필요 없습니다.<br>
```Lat, Long, Location, STREET, REPORTING_AREA```와 같은 위치 정보는 ```DISTRICT``` 만으로 단순화 할 수 있습니다.<br>
```SHOOTING, UCR_PART```는 범죄 예측에 도움이 되지 않는 데이터이므로 필요 없습니다.<br>
```OCCURRED_ON_DATE```는 이미 ```YAER, MONTH, DAY_OF_WEEK, HOUR```로 전처리가 되어있으니 더는 필요 없습니다.<br>

```python
import pandas as pd
# crime.csv에 인코딩이 잘못된 문자가 있기 때문에, engine을 c에서 python으로 바꿔줘야 합니다.
df = pd.read_csv('crime.csv', engine='python')
# 위에서 설명한 필요없는 열들을 지워줍니다.
df = df.drop(columns=['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_DESCRIPTION', 'Lat', 'Long', 'Location', 'STREET', 'REPORTING_AREA', 'SHOOTING', 'UCR_PART', 'OCCURRED_ON_DATE'])
# fixed_crime.csv에 결과물을 적어줍니다.
df.to_csv('fixed_crime.csv', index=False)
```

필요 없는 열들을 모두 지우고 나면 아래 열만 남습니다.<br>

| <center>열 이름</center> | <center>설명</center> |
| OFFENSE_CODE_GROUP | 저지른 범죄가 어떤 종류에 속하는지 알려줍니다. |
| DISTRICT | 어느 지구에서 범죄가 발생했는지 알려줍니다. |
| YEAR | 범죄가 발생한 년도입니다. |
| MONTH | 범죄가 발생한 달입니다. |
| DAY_OF_WEEK | 범죄가 발생한 요일입니다. |
| HOUR | 범죄가 발생한 시간입니다. |

다음으로는 누락된 값이 있는 행을 모두 지워줘야 합니다. 누락된 값이 있으면 제대로 학습이 되지 않을 수 있기 때문입니다.

```python
import pandas as pd

df = pd.read_csv('fixed_crime.csv', engine='python')
df = df.dropna(axis=0)
df.to_csv('fixed_crime2.csv', index=False)
```

이제 간단한 전처리가 모두 끝났습니다! 결과물을 확인해 볼까요?

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fixed_crime2.csv', engine='python')
# OFFENSE_CODE_GROUP에서 각 요소의 갯수를 세서 그래프로 그려줍니다.
df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).plot(kind='barh')
plt.show()
```

![Old Offense Code Group Plot](/img/old_offense_code_group.png)

흠... 뭔가 많습니다. 이대로 바로 사용하는 것은 무리일 것 같습니다.

<br><br><br>

## 데이터 전처리 2

아래 코드를 이용해 OFFENSE_CODE_GROUP에 있는 요소들 목록을 정렬해서 출력할 수 있습니다. 
```python
import pandas as pd

df = pd.read_csv('fixed_crime2.csv', engine='python')
print(*df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist(), sep='\n')
```

<details><summary>출력 결과</summary>
Motor Vehicle Accident Response
Larceny
Medical Assistance
Investigate Person
Other
Drug Violation
Simple Assault
Vandalism
Verbal Disputes
Towed
Investigate Property
Larceny From Motor Vehicle
Property Lost
Warrant Arrests
Aggravated Assault
Violations
Fraud
Residential Burglary
Missing Person Located
Auto Theft
Robbery
Harassment
Property Found
Missing Person Reported
Confidence Games
Police Service Incidents
Disorderly Conduct
Fire Related Reports
Firearm Violations
License Violation
Restraining Order Violations
Counterfeiting
Recovered Stolen Property
Commercial Burglary
Auto Theft Recovery
Liquor Violation
Ballistics
Landlord/Tenant Disputes
Search Warrants
Assembly or Gathering Violations
Property Related Damage
Firearm Discovery
Operating Under the Influence
License Plate Related Incidents
Offenses Against Child / Family
Other Burglary
Evading Fare
Embezzlement
Service
Prisoner Related Incidents
Prostitution
Homicide
Harbor Related Incidents
Criminal Harassment
Arson
HOME INVASION
Bomb Hoax
Aircraft
Phone Call Complaints
Explosives
Gambling
Manslaughter
HUMAN TRAFFICKING
INVESTIGATE PERSON
HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE
Biological Threat
Burglary - No Property Taken
</details>


범죄의 종류가 무려 67개나 됩니다.<br>
일단은 발생한 횟수가 천번 이하인 범죄들을 모두 지워주는 게 좋을 것 같습니다.<br>

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fixed_crime2.csv', engine='python')
value_counts = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).tolist()
value_counts_index = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist()

drop_list = []

for i in range(len(value_counts)):
    if value_counts[i] <= 1000:
        drop_list.append(value_counts_index[i])

df = df[~df['OFFENSE_CODE_GROUP'].isin(drop_list)]

df.to_csv('fixed_crime3.csv', index=False)

df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).plot(kind='barh')
plt.show()
```

![Fixed Offense Code Group](/img/fixed_offense_code_group.png)

36개로 줄이는 데 성공했습니다! 이제 비슷한 범죄끼리 묶어 범죄의 종류를 5~10개 정도로 줄이는 것이 좋겠습니다.<br> 

| OFFENSE_CODE_GROUP | 설명 | 분류 |
| Motor Vehicle Accident Response | 교통사고 |
| Larceny | 절도 |
| Medical Assistance | 자살, 맹견에 물림과 같은 의료 관련 지원 |
| Investigate Person | 사람 조사 |
| Other | 기타 |
| Drug Violation | 약물 |
| Simple Assault | 단순 폭행 |
| Vandalism | 기물파손 |
| Verbal Disputes | 말싸움 |
| Towed | 견인 |
| Investigate Property | 재산 조사 |
| Larceny From Motor Vehicle | 차량 관련 절도 |
| Property Lost | 분실물 |
| Warrant Arrests | 체포 |
| Aggravated Assault | 가중 폭행 |
| Violations | 무단횡단이나 노점상과 같은 법 위반 |
| Fraud | 사기 |
| Residential Burglary | 강도 |
| Missing Person Located | 실종자 발견 |
| Auto Theft | 차량 절도 |
| Robbery | 절도 |
| Harassment | 괴롭힘 |
| Property Found | 부동산 발견(소유자가 없는 부동산 발견)
| Missing Person Reported | 실종자 신고 |
| Confidence Games | 도박 사기 |
| Police Service Incidents | 경찰 서비스 사고 |
| Disorderly Conduct | 무질서한 행동 |
| Fire Related Reports | 화재 관련 |
| Firearm Violations | 총기 위반 |
| License Violation | 저작권 위반 |
| Restraining Order Violations | 접근금지명령 위반 |
| Counterfeiting | 위조 |
| Recovered Stolen Property | 도난 재산 회수 |
| Commercial Burglary | 상점 강도 |
| Auto Theft Recovery | 도난 차량 회수 |
| Liquor Violation | 주류법 위반 |

(분류하고 나니 그래프가 깔끔해졌다는 내용)

## 데이터셋 샘플링

(대충 과적합을 방지하기 위해 under sampling을 한다는 내용)

## 샘플 분할

(대충 6 : 2 : 2 비율로 training set, validation set, test set을 나눈다는 내용)

---

# III. Methodology

- ANN

(인공신경망에 대한 설명)

---

# IV. Evaluation & Analysis

1. 신경망 구조

작성중입니다...

2. 학습 코드

작성중입니다...

3. 학습 결과

(대충 잘 안됐고 이를 해결하기 위해 다른 기법들을 사용해 본다는 내용)

4. 최종 결과

(잘 예측한다는 내용)

---

# V. Related Work
[Crime Type Classification using Neural Networks: A brief walkthrough](https://medium.com/@nicksypark/crime-type-classification-using-neural-networks-a-brief-walkthrough-841b273f9afe)<br>
[Analysis of Boston Crime Incident Open Data Using Pandas](https://towardsdatascience.com/analysis-of-boston-crime-incident-open-data-using-pandas-5ff2fd6e3254)<br>

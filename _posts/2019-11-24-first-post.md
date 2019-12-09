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

## I. Introduction

### 계기

![Minority Report](https://ww.namu.la/s/06a79176babdc3957a67d9adbbc8614f934adea355ce3417d5596f9a236f50a6aa47b9aa4ad9d39ba60f03d8ed5a0dc419a86f9f915af409b6783259cc253aba33c4b3681cf5dc504a62c0f44f6af8131b7aa60c550d5afb0ad94f4addc633c0)  
2002년 개봉한 영화 ‘마이너리티 리포트’는 2054년 워싱턴을 배경으로 합니다.  
주 내용은 범죄가 일어나기 전 범죄를 예측해 범죄자를 단죄하는 최첨단 치안 시스템을 기반으로 하는데요? 이 시스템은 범죄가 일어날 시간과 장소, 범행을 저지를 사람들까지 예측할 수 있고, 특수경찰들이 미래의 범죄자들을 체포합니다.  
우리는 여기서 아이디어를 얻어 2019년 판 ‘피리크라임’을 만들기로 했습니다. 빅데이터와 딥러닝을 통해서 말이죠.  
우리는 2015년부터 2018년까지의 보스턴에서 발생한 범죄 데이터를 바탕으로. 언제 어디서 어떤 범죄가 일어날지 예측하는 시스템을 만들고자 합니다.  

### 최종 목표

4년간의 보스턴의 범죄 데이터를 바탕으로, 앞으로 일어날 범죄를 예측하고 이를 바탕으로 범죄 예방에 초점을 두고 있습니다.  
범죄는 생각보다 불연속적 독립적 성격을 띠고 있지 않습니다. 오히려 연속적이면 비독립적이죠. 서로 연관이 되어 있습니다. 요일에 따라서 날씨에 따라서 범죄의 양뿐만 아니라 범죄의 종류까지 달라집니다.  
저희는 인공신경망을 이용해 4년간의 보스턴의 범죄 데이터를 분석해, 간단한 정보를 넣어줬을 때 어떤 범죄가 발생할 지 예측하는 것을 최종 목표로 선택했습니다.

### 사용할 도구

저희가 사용할 언어는 python 3.6.8 입니다.  
[Anaconda](https://www.anaconda.com/distribution/)라는 프로그램을 사용한다면 여러 파이썬 버전을 충돌 없이 사용할 수 있습니다.  

파이썬 라이브러리로 pytorch, pandas, matplotlib, numpy, tensorboardX(선택) 을 사용할 예정입니다.

```
pip install torch pandas matplotlib numpy --user
```

학습 과정에서의 loss 변화, 정확도 변화 등을 보고 싶다면 tensorboard를 이용해야 합니다.  
각 컴퓨터에 맞는 tensorboard를 설치하신 다음, tensorboardX 라이브러리를 받아 주시면 됩니다.  

```
pip install tensorboardX --user
```

아래 명령어로 설치된 파이썬 버전을 확인할 수 있습니다.  
python 3.6.8가 출력된다면 정상적으로 파이썬이 설치된 것입니다.  

```
python --version
```

아래 명령어로 라이브러리들이 정상적으로 설치되었는지 확인할 수 있습니다.  
아무것도 출력되지 않았다면 라이브러리들이 정상적으로 설치된 것입니다.

```
python -c 'import torch, pandas, matplotlib, numpy'
```
---

## II. Datasets

[캐글 주소](https://www.kaggle.com/AnalyzeBoston/crimes-in-boston)

2015년부터 2018년까지 보스턴에서 발생한 범죄를 통계낸 자료입니다.  
이 자료가 제공하는 csv 파일은 다음과 같습니다.  

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

### 맛보기

이제 이 데이터를 이용해 간단한 그래프를 그려 보겠습니다.  
csv 파일을 모두 다운받고, 파일이 있는 위치로 들어가서 test.py 파일을 만들어 주세요.  

**test.py**
```python
import pandas as pd
import matplotlib.pyplot as plt

# crime.csv에서 데이터를 읽어옵니다. 이 읽어온 데이터를 dataframe 이라고 합니다.
# dataframe에 df라고 이름을 붙여줍니다.
df = pd.read_csv('crime.csv', engine='python')

# dataframe에서 YEAR의 내용을 기준으로 그룹화하고 (groupby)
# 각 그룹의 갯수를 센 다음 (size)
# 막대그래프로 그려줍니다 (plot)
df.groupby('YEAR').size().plot(kind='barh')

# plot 메소드는 내부적으로 matplotlib를 통해 그래프를 그려줍니다.
# 그래서 plt.show()를 통해 그래프가 보이도록 합시다.
plt.show()
```

'YEAR' 부분을 'MONTH', 'DAY_OF_WEEK' 등으로 바꿔서 다른 그래프를 그려볼 수 있습니다.  

![Groupby Year](/aix/img/groupby_year.jpg)

년도를 기준으로 한 그래프입니다. 데이터 수집이 시작된 2015년과 데이터 수집이 끝난 2019년의 범죄수가 적은 것이 보입니다.  
이 두 년도는 다른 년도에 비해 측정 기간이 짧았을 수 밖에 없으니 이건 큰 의미가 없을 것 같습니다.  

![Groupby Month](/aix/img/groupby_month.jpg)

달을 기준으로 한 그래프입니다. 6월~7월이 범죄수가 더 많습니다.  
덥고 습한 여름에 범죄가 많이 일어나는 건 어떻게 보면 당연할 수도 있습니다.  

![Groupby DayOfWeek](/aix/img/groupby_dayofweek.jpg)

요일을 기준으로 한 그래프입니다. 금요일의 범죄수가 가장 많고, 일요일의 범죄수가 가장 적습니다.  

### 데이터 전처리 1

데이터셋을 보면, 불필요한 열이 있습니다.  
```INCIDENT_NUMBER```은 그냥 순서대로 부여되는 의미없는 번호일 뿐이니 필요 없습니다.  
```OFFENSE_CODE, OFFENSE_DESCRIPTION```은 ```OFFENSE_CODE_GROUP``` 만으로도 표현 가능하므로 필요 없습니다.  
```Lat, Long, Location, STREET, REPORTING_AREA```와 같은 위치 정보는 ```DISTRICT``` 만으로 단순화 할 수 있습니다.  
```SHOOTING, UCR_PART```는 범죄 예측에 도움이 되지 않는 데이터이므로 필요 없습니다.  
```OCCURRED_ON_DATE```는 이미 ```YAER, MONTH, DAY_OF_WEEK, HOUR```로 전처리가 되어있으니 더는 필요 없습니다.  

```python
import pandas as pd
# crime.csv에 인코딩이 잘못된 문자가 있기 때문에, engine을 c에서 python으로 바꿔줘야 합니다.
df = pd.read_csv('crime.csv', engine='python')
# 위에서 설명한 필요없는 열들을 지워줍니다.
df = df.drop(columns=['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_DESCRIPTION', 'Lat', 'Long', 'Location', 'STREET', 'REPORTING_AREA', 'SHOOTING', 'UCR_PART', 'OCCURRED_ON_DATE'])
# fixed_crime.csv에 결과물을 적어줍니다.
df.to_csv('fixed_crime.csv', index=False)
```

필요 없는 열들을 모두 지우고 나면 아래 열만 남습니다.  

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

![Old Offense Code Group Plot](/aix/img/old_offense_code_group.png)

흠... 뭔가 많습니다. 이대로 바로 사용하는 것은 무리일 것 같습니다.  

### 데이터 전처리 2

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

범죄의 종류가 무려 67개나 됩니다.  
일단은 발생한 횟수가 천번 이하인 범죄들을 모두 지워주는 게 좋을 것 같습니다.  

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fixed_crime2.csv', engine='python')
value_counts = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).tolist()
value_counts_index = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist()

other_list = []

for i in range(len(value_counts)):
    if value_counts[i] <= 1000:
        other_list.append(value_counts_index[i])

def func(x):
    if x in other_list:
        return 'Other'
    else:
        return x

df['OFFENSE_CODE_GROUP'] = df.apply(lambda x: func(x['OFFENSE_CODE_GROUP']), axis=1)

df.to_csv('fixed_crime3.csv', index=False)

df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).plot(kind='barh')
plt.show()
```

![Fixed Offense Code Group](/aix/img/fixed_offense_code_group.png)

36개로 줄이는 데 성공했습니다! 이제 비슷한 범죄끼리 묶어 범죄의 종류를 5~10개 정도로 줄이는 것이 좋겠습니다.  

| OFFENSE_CODE_GROUP | 설명 | 분류 |
| Motor Vehicle Accident Response | 교통사고 | 사고 |
| Larceny | 절도 | 절도 |
| Medical Assistance | 자살, 맹견에 물림과 같은 사고 처리 | 사고 |
| Investigate Person | 사람 조사 | 서비스 |
| Other | 기타 | 기타 |
| Drug Violation | 약물 | 경범죄 |
| Simple Assault | 단순 폭행 | 폭행 |
| Vandalism | 기물파손 | 경범죄 |
| Verbal Disputes | 말싸움 | 폭행 |
| Towed | 견인 | 경범죄 |
| Investigate Property | 재산 조사 | 서비스 |
| Larceny From Motor Vehicle | 차량 관련 절도 | 절도 |
| Property Lost | 분실물 | 절도 |
| Warrant Arrests | 체포 | 서비스 |
| Aggravated Assault | 가중 폭행 | 폭행 |
| Violations | 무단횡단이나 노점상과 같은 법 위반 | 경범죄 |
| Fraud | 사기 | 절도 |
| Residential Burglary | 강도 | 절도 |
| Missing Person Located | 실종자 발견 | 서비스 |
| Auto Theft | 차량 절도 | 절도 |
| Robbery | 강도 | 절도 |
| Harassment | 괴롭힘 | 폭행 |
| Property Found | 부동산 발견(소유자가 없는 부동산 발견) | 서비스 |
| Missing Person Reported | 실종자 신고 | 서비스 |
| Confidence Games | 도박 사기 | 절도 |
| Police Service Incidents | 경찰 서비스 사고 | 사고 |
| Disorderly Conduct | 무질서한 행동 | 경범죄 |
| Fire Related Reports | 화재 관련 | 사고 |
| Firearm Violations | 총기 위반 | 경범죄 |
| License Violation | 저작권 위반 | 경범죄 |
| Restraining Order Violations | 접근금지명령 위반 | 경범죄 |
| Counterfeiting | 위조 | 경범죄 |
| Recovered Stolen Property | 도난 재산 회수 | 서비스 |
| Commercial Burglary | 상점 강도 | 절도 |
| Auto Theft Recovery | 도난 차량 회수 | 절도 |
| Liquor Violation | 주류법 위반 | 경범죄 |

```python
import pandas as pd
import matplotlib.pyplot as plt

accident_list = ['Medical Assistance', 'Motor Vehicle Accident Response', 'Fire Related Reports', 'Police Service Incidents']
theft_list = ['Larceny', 'Larceny From Motor Vehicle', 'Property Lost', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Fraud', 'Confidence Games', 'Commercial Burglary', 'Auto Theft Recovery']
misdemeanor_list = ['Drug Violation', 'Liquor Violation', 'Vandalism', 'Towed', 'Violations', 'Disorderly Conduct', 'Firearm Violations', 'License Violation', 'Restraining Order Violations', 'Counterfeiting']
service_list = ['Investigate Person', 'Investigate Property', 'Warrant Arrests', 'Missing Person Located', 'Property Found', 'Missing Person Reported', 'Recovered Stolen Property']
violence_list = ['Verbal Disputes', 'Simple Assault', 'Aggravated Assault', 'Harassment']
other_list = ['Other']

df = pd.read_csv('fixed_crime3.csv', engine='python')

def func(x):
    if x in accident_list:
        return 'Accident'
    if x in theft_list:
        return 'Theft'
    if x in misdemeanor_list:
        return 'Misdemeanor'
    if x in service_list:
        return 'Service'
    if x in violence_list:
        return 'Violence'
    if x in other_list:
        return 'Other'

df['CLASSIFICATION'] = df.apply(lambda x: func(x['OFFENSE_CODE_GROUP']), axis=1)

df.to_csv('fixed_crime4.csv', index=False)

df['CLASSIFICATION'].value_counts(sort=True, dropna=False).plot(kind='barh')
plt.show()
```

![Last Offense Code Group](/aix/img/last_offense_code_group.png)

6개로 줄이는 데 성공했습니다! 데이터의 불균형이 많이 해결된 것이 보입니다.  

### 데이터셋 샘플링

![Last Offense Code Group](/aix/img/last_offense_code_group.png)

위 그래프를 보면, Theft가 Other보다 2배 정도 많습니다.  
이렇게 데이터 불균형(imbalanced data)가 있으면, 학습이 제대로 안 됩니다.  
예를 들어, 데이터 10만개 중 9만개가 a클래스이고, 1만개가 b클래스라면 항상 a클래스라고 데이터를 분류하는 모델의 정확도가 90%가 되어 버립니다.  
그래서 보통 머신 러닝 알고리즘들은 데이터셋에서 각 클래스의 개수가 동일할 때 좋은 성능을 보여줍니다.  

이 문제를 해결하기 위해, Random undersampling이라는 기법을 사용할 것입니다.  
Random undersampling은 간단합니다. 그냥 가장 적은 개수의 클래스(저희의 경우 Other)의 개수가 되도록 다른 클래스들의 데이터들을 무작위로 지워 주면 됩니다.  

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('fixed_crime4.csv', engine='python')

drop_list = ['Accident', 'Theft', 'Misdemeanor', 'Service', 'Violence']

new_df = df[df['CLASSIFICATION'] == 'Other']
other_count = len(df[df['CLASSIFICATION'] == 'Other'])

for class_name in drop_list:
    class_index = df[df['CLASSIFICATION'] == class_name].index
    under_sample_index = np.random.choice(class_index, other_count, replace=False)
    under_sample = df.loc[under_sample_index]
    new_df = new_df.append(under_sample)

new_df = new_df.sample(frac=1).reset_index(drop=True)

new_df.to_csv('fixed_crime5.csv', index=False)

new_df['CLASSIFICATION'].value_counts(sort=True, dropna=False).plot(kind='barh')
plt.show()
```

![Undersampling Result](/aix/img/undersampling_result.png)

### 데이터 추가 / 삭제

이대로 바로 학습을 진행하는 것도 좋겠지만, 데이터를 조금 더 추가해줍시다.  
범죄가 발생한 시간대가 Morning, Afternoon, Evening, Night 중 어디에 속하는지, 범죄가 발생한 날이 weekday인지 아닌지를 추가해주겠습니다.  
또한, 인적 피해, 재산적 피해가 있었는지를 추가하고, 건물 내부에서 발생하는 범죄인지, 피해자와 가해자가 명확하게 존재하는 범죄인지, 차량에 관련된 것인지, 죄인이 존재하는지등을 추가로 더 넣어주겠습니다.  

| OFFENSE_CODE_GROUP | 인적 피해 | 재산적 피해 | 건물 내부 | 피해자, 가해자 명확 | 차량 관련 | 죄인 존재 |
| Motor Vehicle Accident Response | O | O | X | X | O | X |
| Larceny | X | O | X | O | X | O |
| Medical Assistance | O | X | X | X | X | X |
| Investigate Person | X | X | X | X | X | X |
| Other | X | X | X | X | X | X |
| Drug Violation | O | X | X | X | X | O |
| Simple Assault | O | X | X | O | X | O |
| Vandalism | X | O | O | X | X | O |
| Verbal Disputes | O | X | X | O | X | O |
| Towed | X | X | X | X | O | O |
| Investigate Property | X | X | X | X | X | X |
| Larceny From Motor Vehicle | X | O | X | O | O | O |
| Property Lost | X | O | X | O | X | X |
| Warrant Arrests | X | X | X | X | X | O |
| Aggravated Assault | O | X | X | O | X | O |
| Violations | X | X | X | X | X | O |
| Fraud | X | O | X | O | X | O |
| Residential Burglary | O | O | X | O | X | O |
| Missing Person Located | X | X | X | X | X | X |
| Auto Theft | X | O | X | O | O | O |
| Robbery | O | O | X | O | X | O |
| Harassment | O | X | X | O | X | O |
| Property Found | X | X | X | X | X | X |
| Missing Person Reported | X | X | X | X | X | X |
| Confidence Games | X | O | X | O | X | O |
| Police Service Incidents | X | X | X | X | X | X |
| Disorderly Conduct | X | X | X | X | X | O |
| Fire Related Reports | X | O | X | X | X | X |
| Firearm Violations | X | X | X | X | X | O |
| License Violation | X | X | X | O | X | O |
| Restraining Order Violations | X | X | X | O | X | O |
| Counterfeiting | X | O | X | X | X | O |
| Recovered Stolen Property | X | X | X | X | X | X |
| Commercial Burglary | X | O | O | O | X | O |
| Auto Theft Recovery | X | O | X | O | O | O |
| Liquor Violation | X | X | X | X | X | O |

데이터를 추가한 다음 더는 필요 없는 OFFENSE_CODE_GROUP를 지워줍시다.

```python
import pandas as pd

df = pd.read_csv('fixed_crime5.csv', engine='python')

def func_time(x):
    if 6 <= x and x < 12:
        return 'Morning'
    if 12 <= x and x < 17:
        return 'Afternoon'
    if 5 <= x and x < 19:
        return 'Evening'
    return 'Night'

df['TIME'] = df.apply(lambda x: func_time(x['HOUR']), axis=1)

def func_weekday(x):
    if x == 'Sunday' or x == 'Saturday':
        return 1
    else:
        return 0

df['WEEKDAY'] = df.apply(lambda x: func_weekday(x['DAY_OF_WEEK']), axis=1)

human_loss = ['Motor Vehicle Accident Response', 'Medical Assistance', 'Drug Violation', 'Simple Assault', 'Verbal Disputes', 'Aggravated Assault', 'Residential Burglary', 'Robbery', 'Harassment']
financial_loss = ['Motor Vehicle Accident Response', 'Larceny', 'Vandalism', 'Larceny From Motor Vehicle', 'Property Lost', 'Fraud', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Confidence Games', 'Fire Related Reports', 'Counterfeiting', 'Commercial Burglary', 'Auto Theft Recovery']
inside = ['Vandalism', 'Commercial Burglary']
exist_victim_offender = ['Larceny', 'Simple Assault', 'Verbal Disputes', 'Larceny From Motor Vehicle', 'Property Lost', 'Aggravated Assault', 'Fraud', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Harassment', 'Confidence Games', 'License Violation', 'Commercial Burglary', 'Auto Theft Recovery']
about_auto = ['Motor Vehicle Accident Response', 'Towed', 'Larceny From Motor Vehicle', 'Auto Theft', 'Auto Theft Recovery']
exist_sinner = ['Larceny', 'Drug Violation', 'Simple Assault', 'Vandalism', 'Verbal Disputes', 'Towed', 'Larceny From Motor Vehicle', 'Warrant Arrests', 'Aggravated Assault', 'Violations', 'Fraud', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Harassment', 'Confidence Games', 'Disorderly Conduct', 'Firearm Violations', 'License Violation', 'Restraining Order Violations', 'Counterfeiting', 'Commercial Burglary', 'Auto Theft Recovery', 'Liquor Violation']

def func_in_list(x, lst):
    if x in lst:
        return 1
    else:
        return 0

df['HUMAN_LOSS'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], human_loss), axis=1)
df['FINANCIAL_LOSS'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], financial_loss), axis=1)
df['INSIDE'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], inside), axis=1)
df['EXIST_VICTIM_OFFENDER'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], exist_victim_offender), axis=1)
df['ABOUT_AUTO'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], about_auto), axis=1)
df['EXIST_SINNER'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], exist_sinner), axis=1)

del df['OFFENSE_CODE_GROUP']

df.to_csv('fixed_crime6.csv', index=False)
```

### 샘플 분할

fixed_crim6.csv의 내용을 8 : 1 : 1의 비율로 training set, validation set, test set으로 나눠주겠습니다.

```python
import pandas as pd

df = pd.read_csv('fixed_crime6.csv', engine='python')

classes = ['Accident', 'Other', 'Misdemeanor', 'Violence', 'Service', 'Theft']

dfs = {}
for c in classes:
    dfs[c] = df[df['CLASSIFICATION'] == c]

training_set = []
validation_set = []
test_set = []

for c in classes:
    train = dfs[c].sample(frac=0.8)
    dfs[c] = dfs[c].drop(train.index)

    validation = dfs[c].sample(frac=0.5)
    dfs[c] = dfs[c].drop(validation.index)

    training_set.append(train)
    validation_set.append(validation)
    test_set.append(dfs[c])

training_set_df = training_set[0]
validation_set_df = validation_set[0]
test_set_df = test_set[0]

for i in range(1, len(training_set)):
    training_set_df = training_set_df.append(training_set[i])
    validation_set_df = validation_set_df.append(validation_set[i])
    test_set_df = test_set_df.append(test_set[i])

training_set_df = training_set_df.sample(frac=1).reset_index(drop=True)
validation_set_df = validation_set_df.sample(frac=1).reset_index(drop=True)
test_set_df = test_set_df.sample(frac=1).reset_index(drop=True)

training_set_df.to_csv('training_set.csv', index=False)
validation_set_df.to_csv('validation_set.csv', index=False)
test_set_df.to_csv('test_set.csv', index=False)
```

### one-hot encoding

```python
import pandas as pd

test_set_df = pd.read_csv('test_set.csv', engine='python')
training_set_df = pd.read_csv('training_set.csv', engine='python')
validation_set_df = pd.read_csv('validation_set.csv', engine='python')

test_set_df = pd.get_dummies(test_set_df, columns=test_set_df.columns)
training_set_df = pd.get_dummies(
    training_set_df, columns=training_set_df.columns)
validation_set_df = pd.get_dummies(
    validation_set_df, columns=validation_set_df.columns)

test_set_df.to_csv('test_set_one_hot.csv', index=False)
training_set_df.to_csv('training_set_one_hot.csv', index=False)
validation_set_df.to_csv('validation_set_one_hot.csv', index=False)
```

---

## III. Methodology

- ANN

(인공신경망에 대한 설명)

---

## IV. Evaluation & Analysis

### 신경망 구조

![DNN](https://t1.daumcdn.net/cfile/tistory/212B724858BCE20914)

저희가 사용할 신경망은 위 그림과 비슷하게 생긴 Deep Neural Network 입니다.

| name | activation function | input | output |
| input_layer | ReLu | 77 | 256 |
| hidden_layer_1 | ReLu | 256 | 128 |
| hidden_layer_2 | ReLu | 128 | 128 |
| hidden_layer_3 | ReLu | 128 | 128 |
| output_layer | Softmax | 128 | 6 |

input_layer에 one-hot encoding 된 input을 넣어주면, output_layer에서 6개의 범죄 분류에 속할 확률을 출력해 주는 신경망입니다.

### 학습 코드

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

from tensorboardX import SummaryWriter
writer = SummaryWriter()

batch_size = 100

class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()

        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer_1 = nn.Linear(256, 128)
        self.hidden_layer_2 = nn.Linear(128, 128)
        self.hidden_layer_3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.relu(self.hidden_layer_3(x))
        x = F.softmax(self.output_layer(x))
        return x

target = ['CLASSIFICATION_Accident', 'CLASSIFICATION_Misdemeanor', 'CLASSIFICATION_Other', 'CLASSIFICATION_Service', 'CLASSIFICATION_Theft', 'CLASSIFICATION_Violence']

training_set_df = pd.read_csv('training_set_one_hot.csv', engine='python')
training_set_target_df = training_set_df[target]
training_set_df = training_set_df.drop(columns=target)
training = TensorDataset(torch.from_numpy(np.array(training_set_df)), torch.from_numpy(np.array(training_set_target_df)))
training_loader = DataLoader(training, batch_size=batch_size, shuffle=True)

validation_set_df = pd.read_csv('validation_set_one_hot.csv', engine='python')
validation_set_target_df = validation_set_df[target]
validation_set_df = validation_set_df.drop(columns=target)
validation = TensorDataset(torch.from_numpy(np.array(validation_set_df)), torch.from_numpy(np.array(validation_set_target_df)))
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

test_set_df = pd.read_csv('test_set_one_hot.csv', engine='python')
test_set_target_df = test_set_df[target]
test_set_df = test_set_df.drop(columns=target)
test = TensorDataset(torch.from_numpy(np.array(test_set_df)), torch.from_numpy(np.array(test_set_target_df)))
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

print(len(training_set_df.columns))
model = Network(len(training_set_df.columns))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100

valid_loss_min = np.Inf

for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()

    train_count = 0
    valid_count = 0

    for data, target in training_loader:
        optimizer.zero_grad()
        output = model(data.float())
        loss = criterion(output.float(), target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        train_count += 1

    model.eval()
    for data, target in validation_loader:
        output = model(data.float())
        loss = criterion(output.float(), target.float())
        valid_loss += loss.item()*data.size(0)
        valid_count += 1

    train_loss = train_loss / train_count
    valid_loss = valid_loss / valid_count

    writer.add_scalars('loss/train+valid', {'train': train_loss, 'valid': valid_loss}, epoch)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


model.eval()
test_loss=0
correct=0
for data,target in test_loader:
    output = model(data.float())
    for i in range(len(target)):
        if target[i][output[i].max(0)[1]] == 1:
            correct += 1

print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

```

### 1차 학습 결과

![Overfitting Graph](/aix/img/overfitting.png)

빨간색이 training set의 loss 그래프, 파란색이 validation set의 loss 그래프입니다.  
training set의 loss는 epoch가 지나며 계속 감소하지만, 오히려 validation set의 loss 그래프는 계속 증가하는 것이 보입니다.

![Overfitting](https://pbs.twimg.com/media/CVz1XohXIAAW6PQ.jpg)

그렇습니다. overfitting이 발생했습니다.  
모델이 training set에 overfitting 되었기 때문에 validation set으로 테스트 했을 때의 오차가 줄어들지 않고 계속 증가하는 것입니다.  
overfitting을 줄이기 위해 dropout이라는 기법을 이용하겠습니다.

### Dropout 적용

pytorch에서 dropout을 적용하는 것은 생각보다 간단합니다. Network 클래스를 만들 때, nn.Dropout 함수를 이용해서 dropout 레이어를 만들 수 있습니다.  

```python
class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()

        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer_1 = nn.Linear(256, 128)
        self.hidden_layer_2 = nn.Linear(128, 128)
        self.hidden_layer_3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 6)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)
        self.dropout4 = nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden_layer_1(x))
        x = self.dropout2(x)
        x = F.relu(self.hidden_layer_2(x))
        x = self.dropout3(x)
        x = F.relu(self.hidden_layer_3(x))
        x = self.dropout4(x)
        x = F.softmax(self.output_layer(x))
        return x
```


### 2차 학습 결과

![Droupout](/aix/img/dropout.png)

### 최종 결과

![Last Result](/aix/img/last_result.png)

---

# V. Related Work
[Crime Type Classification using Neural Networks: A brief walkthrough](https://medium.com/@nicksypark/crime-type-classification-using-neural-networks-a-brief-walkthrough-841b273f9afe)  
[Analysis of Boston Crime Incident Open Data Using Pandas](https://towardsdatascience.com/analysis-of-boston-crime-incident-open-data-using-pandas-5ff2fd6e3254) 

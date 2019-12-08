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
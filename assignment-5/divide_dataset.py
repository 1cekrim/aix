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
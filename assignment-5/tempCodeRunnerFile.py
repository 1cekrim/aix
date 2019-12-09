import pandas as pd

test_set_df = pd.read_csv('test_set.csv', engine='python')
training_set_df = pd.read_csv('training_set.csv', engine='python')
validation_set_df = pd.read_csv('validation_set.csv', engine='python')

test_set_df = pd.get_dummies(test_set_df, columns=test_set_df.columns)
training_set_df = pd.get_dummies(training_set_df, columns=training_set_df.columns)
validation_set_df = pd.get_dummies(validation_set_df, columns=validation_set_df.columns)

test_set_df.to_csv('test_set_one_hot.csv', index=False)
training_set_df.to_csv('training_set_one_hot.csv', index=False)
validation_set_df.to_csv('validation_set_one_hot.csv', index=False)
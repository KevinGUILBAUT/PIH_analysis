import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Make sure it's the full data for the both datasets
df1 = pd.read_csv('./data/features_extracted/data.csv')
df2 = pd.read_csv('./data/features_extracted/data_chu.csv')

df = pd.concat([df1, df2], ignore_index=True)
df.to_csv('./data/features_extracted/data_vitaldb_chu.csv')

X = df.drop(columns='label')
y = df['label']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3,stratify=y)

train = X_train.copy()
train['label'] = y_train
test = pd.concat([X_test, y_test], axis=1)

nb_cv = 3
skf = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=42)

cv_split_col = pd.Series(index=train.index, dtype="object")
for i, (_, val_idx) in enumerate(skf.split(train.drop(columns='label'), train['label'])):
    cv_split_col.iloc[val_idx] = f'cv_{i}'

train['cv_split'] = cv_split_col
test['cv_split'] = 'test'

test.to_csv("./data/features_extracted/test_vitaldb_chu.csv", index=False)
train.to_csv("./data/features_extracted/train_vitaldb_chu.csv", index=False)

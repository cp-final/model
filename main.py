import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import *
import fasttext 
import fasttext.util


df = pd.read_csv('200k_train.csv')


cat_cols = [
    'created',
    'shift',
    'oblast',
    'city',
    'os',
    'gamecategory',
    'subgamecategory'
]

def convCol(df1, name):
  le = LabelEncoder()
  le.fit(df1[name])
  df1[name]=le.transform(df1[name])
  return df1

def convert_android(df):
  df.loc[df['os']==0, 'bundle'] = df.loc[(df['os']==0)]['bundle'].str.strip('com.')
  df.loc[df['os']==0, 'bundle'] = df.loc[(df['os']==0)]['bundle'].str.replace(".", " ")
  return df

def clear_csv(df):
    df.drop(['shift'], axis=1)
    df['os'] = df['os'].str.lower()
    df.loc[df['os']=='Android', 'os'] = 'android'
    df.loc[df['os']=='Ios', 'os'] = 'ios'

    for i, date in enumerate(df['created']):
      time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
      df.loc[i, 'created'] = int(time.hour)

    for name in cat_cols:
      if name!='created':
        df = convCol(df, name)
    df.drop('created', axis=1)
    return convert_android(df)


dfg = clear_csv(df)

dfg.to_csv('/content/drive/MyDrive/dataset/please-convert.csv', index=False)



ft = fasttext.load_model('/content/drive/MyDrive/dataset/cc.en.300.bin')
ft.get_dimension()

def process(dff, model):
  dff.loc[dff['bundle']!='Nan', 'bundles'] = model.predict(dff.loc[(dff['bundle']!='Nan')]['bundle'].str)
  
  return dff

def apple_convert(dff, fg):
  dff.loc[dff['bundle']!='Nan', 'bundle'] = fg(dff.loc[(dff['boundle']!='Nan')]['bundle'])
  return dff

g = process(g, ft)
g.head(10)

#New Block code

fdf = pd.read_csv('/content/drive/MyDrive/dataset/final_train.csv')

fdf.sample(30)

fdf.sample(30)# new

fdf = fdf.drop('osv', axis=1)

#fdf = fdf.drop('Unnamed: 0', axis=1)
fdf = fdf.dropna()

fdf.info()

for i, item in enumerate(fdf['bundle']):
  if type(item) !=float:
    fdf['bundle'][i] = np.fromstring(item[1:len(item)-1], dtype=float, sep=' ')
  else: 
    fdf['bundle'][i] = np.array[np.NaN]
fdf.info()

fdf = fdf_

for i in fdf()

fdf.sample(40)

fdf = fdf.dropna()

fdf.sample(20)

def create_new_column(df):
  fg = pd.DataFrame(np.zeros((len(df), 30)))
  for i, item in enumerate(df['bundle']):
    print(i)
    if len(item)>10:
      for j in range(len(item)):
        fg.loc[i, j] = item[j]
          

  #print(fg.head())
  #print(fg.info())
  return fg

clear_vector_df = create_new_column(fdf)
clear_vector_df.head()

predata = pd.concat([fdf, clear_vector_df], axis=1)

predata = predata.drop('bundle', axis=1)
predata = predata.dropna()

predata

sav = predata
predata = predata.drop('Unnamed: 0', axis=1)
predata.head()

predata.to_csv('/content/drive/MyDrive/dataset/predata2.csv', index=False)

y = predata['Segment']

X = predata.drop(['Segment'], axis=1)
X.head()

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

X.head()

copy_X = X

X['created'] = fdf['created']

X.head()

ans = pd.concat([X, y], axis=1)
ans.head()

ans = ans.dropna()

ans.to_csv('/content/drive/MyDrive/dataset/alast.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_train.sample(20)

X_train.sample()

y_train

#check have nan
is_NaN = ans.isnull()
row_has_NaN = is_NaN.any(axis=1)
print(np.unique(row_has_NaN))
rows_with_NaN = df[row_has_NaN]

print(rows_with_NaN)

np.all(np.isfinite(ans))

new_dataframe = ans.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

new_dataframe.head()

Y = new_dataframe['Segment']

X = new_dataframe.drop('Segment', axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

kmeans = KMeans(n_clusters=5, random_state=58, max_iter=10000, algorithm='elkan')
print(kmeans.fit_predict(X_train, y_train))

ans_model_kmean = kmeans.predict(X_test)
print(ans_model_kmean)

print(y_test)



print(len(y_test))
y_test_np = y_test.to_numpy()
print(type(y_test_np))
print(len(ans_model_kmean))
print(type(ans_model_kmean))

score = 0
for i in range(len(ans_model_kmean)):
  if y_test_np[i] == ans_model_kmean[i]:
    score+=1
print(score/len(ans_model_kmean))

#К средних не отвечает сложности задачи.

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, max_iter=300, verbose=1, solver = 'adam', learning_rate='adaptive', hidden_layer_sizes=(120), activation='tanh').fit(X_train, y_train)

predict_clf = clf.predict(X_test)

print(len(y_test))
y_test_np = y_test.to_numpy()
print(type(y_test_np))
print(len(predict_clf))
print(type(predict_clf))

predict_clf

score = 0
for i in range(len(predict_clf)):
  if y_test_np[i] == predict_clf[i]:
    score+=1
print(score/len(predict_clf))

import pickle

filename = '/content/drive/MyDrive/dataset/model0.64.mdl'
pickle.dump(clf, open(filename, 'wb'))

from sklearn.metrics import roc_auc_score

y_test = y_test.to_numpy()

type(y_test)

type(predict_clf)

roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')


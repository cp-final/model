import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import *
import fasttext 
import fasttext.util
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from multiprocessing.pool import ThreadPool


df = pd.read_csv('200k_train.csv')[:50]
print("Reading CSV")


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

def normalize_time(date, df):
  time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
  df.loc[i, 'created'] = int(time.hour)
  print("change time")


def clear_csv(df):
    print("Stage 1")
    df.drop(['shift'], axis=1)
    df['os'] = df['os'].str.lower()
    df.loc[df['os']=='Android', 'os'] = 'android'
    df.loc[df['os']=='Ios', 'os'] = 'ios'
    
    print("Stage 2 spliting time")
    for i, item in enumerate(df['created']):
      time = datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
      df.loc[i, 'created'] = int(time.hour)



    print("Stage 3 encoding")
    for name in cat_cols:
      if name!='created':
        df = convCol(df, name)
    df.drop('created', axis=1)
    return convert_android(df)


print("Start clearing CSV")
dfg = clear_csv(df)
print("End clearing CSV")


ft_model = fasttext.load_model('cc.en.300.bin')

def process(dff, model):
  data = []
  for i, item in enumerate(dff['bundle']):
      if type(item) == str:
          vector_word = model.get_word_vector(str(item).lower())
          data.append(vector_word)
      else:
          data.append(np.nan)
  dff['bundle'] = data
  return dff

def apple_convert(dff, fg):
  dff.loc[dff['bundle']!='Nan', 'bundle'] = fg(dff.loc[(dff['boundle']!='Nan')]['bundle'])
  return dff

vectorized_df = process(dfg, ft_model)

vectorized_df = vectorized_df.drop('osv', axis=1)
vectorized_df = vectorized_df.dropna()

print("Dataframe vectorized")


#for i, item in enumerate(fdf['bundle']):
#  if type(item) !=float:
#    fdf['bundle'][i] = np.fromstring(item[1:len(item)-1], dtype=float, sep=' ')
#  else: 
#    fdf['bundle'][i] = np.array[np.NaN]
#fdf.info()

def create_new_column(df):
  fg = pd.DataFrame(np.zeros((len(df), 30)))
  for i, item in enumerate(df['bundle']):
    print(i)
    if len(item)>10:
      for j in range(len(item)):
        fg.loc[i, j] = item[j]
  return fg

clear_vector_df = create_new_column(vectorized_df)
predata = pd.concat([vectorized_df, clear_vector_df], axis=1)
predata = predata.drop('bundle', axis=1)
predata = predata.dropna()

predata.to_csv('predata2.csv', index=False)
print("Saved predata")

y = predata['Segment']
X = predata.drop(['Segment'], axis=1)

copy_X = X
X['created'] = vectorized_df['created']
ans = pd.concat([X, y], axis=1)
ans = ans.dropna()

ans.to_csv('alast.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

is_NaN = ans.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = vectorized_df[row_has_NaN]

new_dataframe = ans.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

Y = new_dataframe['Segment']
X = new_dataframe.drop('Segment', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

print("Start learning NN")
kmeans = KMeans(n_clusters=5, random_state=58, max_iter=10000, algorithm='elkan')
print(kmeans.fit_predict(X_train, y_train))

#ans_model_kmean = kmeans.predict(X_test)
#print(ans_model_kmean)
#
#print(y_test)
#
#
#
#print(len(y_test))
#y_test_np = y_test.to_numpy()
#print(type(y_test_np))
#print(len(ans_model_kmean))
#print(type(ans_model_kmean))
#
#score = 0
#for i in range(len(ans_model_kmean)):
#  if y_test_np[i] == ans_model_kmean[i]:
#    score+=1
#print(score/len(ans_model_kmean))
#
##К средних не отвечает сложности задачи.
#
#from sklearn.neural_network import MLPClassifier
#
#clf = MLPClassifier(random_state=1, max_iter=300, verbose=1, solver = 'adam', learning_rate='adaptive', hidden_layer_sizes=(120), activation='tanh').fit(X_train, y_train)
#
#predict_clf = clf.predict(X_test)
#
#print(len(y_test))
#y_test_np = y_test.to_numpy()
#print(type(y_test_np))
#print(len(predict_clf))
#print(type(predict_clf))
#
#predict_clf
#
#score = 0
#for i in range(len(predict_clf)):
#  if y_test_np[i] == predict_clf[i]:
#    score+=1
#print(score/len(predict_clf))
#
#import pickle
#
#filename = '/content/drive/MyDrive/dataset/model0.64.mdl'
#pickle.dump(clf, open(filename, 'wb'))
#
#from sklearn.metrics import roc_auc_score
#
#y_test = y_test.to_numpy()
#
#type(y_test)
#
#type(predict_clf)
#
#roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
#
#roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
#

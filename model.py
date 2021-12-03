from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from datetime import *
import fasttext 
import fasttext.util
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle
import sys


loaded_model = pickle.load(open('models/model_450k_240_mlp.mdl', 'rb'))

csv_name = sys.argv[1]

df = pd.read_csv(csv_name)
df_copy = df.copy()
apps_df = pd.read_csv('apps_id.csv')
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
    
def get_ios_app(app_id):
    return apps_df[apps_df['id'] == app_id]['app']

def normalize_ios(df):
     
    pass


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
      if df['os'][i] == "ios":
          app_name = get_ios_app(df['bundle'][i])
          print(app_name)
          if not app_name.empty:
              df.loc[i, 'bundle'] = str(app_name)

    


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
fasttext.util.reduce_model(ft_model, 30)

def process(dff, model):
    data = []
    for i, item in enumerate(dff['bundle']):
        if type(item) != float:
            if not item.isnumeric():
                if type(item) == str:
                    vector_word = model.get_word_vector(str(item).lower())
                    data.append(vector_word)
                else:
                    data.append(np.nan)
            else:
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

print(predata)
print("Saved predata")


#is_NaN = ans.isnull()
#row_has_NaN = is_NaN.any(axis=1)
#rows_with_NaN = vectorized_df[row_has_NaN]

new_dataframe = predata.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


res = loaded_model.predict(new_dataframe)
res = pd.DataFrame({'segment':res})
dd = pd.concat([res, df_copy],axis=1)
dd.to_csv('new_prediction_test.csv', index=None)

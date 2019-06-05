import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2.5)
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline

# 1 dataset confirmation(dataset의 column과 column에 대한 통계적인 수치를 확인한다.)
print("1. dataset confirmation")

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print(df_train.head())
print(df_train.describe())
print(df_test.describe())

# 1.1 null data check(null data가 어떻게 분포하고 있는지 확인한다.)
print("1.1 null data")

for col in df_train.columns:
    msg = 'column: {:>11}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_train[col].isnull().sum()/df_train[col].shape[0]))
    print(msg)
print()

for col in df_test.columns:
    msg = 'column: {:>11}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_test[col].isnull().sum()/df_test[col].shape[0]))
    print(msg)

plt.show(msno.matrix(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8,0.5,0.2)))
plt.show(msno.bar(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8,0.5,0.2)))

# 1.2 Target Label Confirmation (label의 분포를 확인한다.)
print("1.2 Target Label Confirmation")

f, ax = plt.subplots(1,2,figsize=(18,8))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')
plt.show()

# 2. Exploratory data analysis
print("2. Exploratory data analysis")

# 2.1 Pclass
print("Pclass")
# df_train[['Pclass','Survived']]
# pclass
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count())
# 각 pclass 마다 생존한 사람(Survived=1)
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum())
# 각 class 마다 사망한 사람 수, 생존한 사람 수, 모든 사람 수에 대한 테이블
print(pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True))
# class별 생존률 시각화
plt.show(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar())

y_position = 1.02
f, ax = plt.subplots(1,2,figsize=(18,8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()
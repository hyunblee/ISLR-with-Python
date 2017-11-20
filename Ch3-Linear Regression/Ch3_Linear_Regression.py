
# coding: utf-8

# # Chapter 3 - Linear Regression
# 
# ### ISLR 3장의 Linear Regression Python 실습 
# 
# ## Linear Regression (선형 회귀분석)##
# - Response의 값이 숫자인 labeled 데이타를 이용하는 **Regression** 타입 **Supervised Learning** 모델
# - 빨리 돌고, 오랜 시간 많이 연구되어 특성을 잘 알고, 모델의 해석이 쉬워 널리 사용
# 
# ### 사용할 주요 Python 패키지
# - [pandas](http://pandas.pydata.org)  : 데이터 입출력, Munging, & etc.
# - [numpy](http://www.numpy.org/)  : 수식 계산 
# - [matplotlib](http://matplotlib.org/)  : 시각화 
# - [seaborn](https://seaborn.github.io/index.html)  : 시각화 
# - **[statsmodels](http://statsmodels.sourceforge.net/)  : 통계모델**
# - **[scikit-learn](http://scikit-learn.org/stable)  : 머신러닝** 

# ### * [Statsmodels](http://statsmodels.sourceforge.net/) 패키지의 모델을 사용해 Linear Regression을 익힌다.*  ###
# 
# Statsmodels 의 Linear Regression 모델은 ISLR 책의 R 쓰임새와 비슷하게 사용할 수 있음
# 
# - **[R 스타일 formula로 모델 만들기](http://statsmodels.sourceforge.net/stable/example_formulas.html)**

# In[1]:

# 패키지 imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf     # R의 formula 식 유사하게 쓰임

from hblee import st,Corrplot    # hblee.py: 웹에서 훔쳤거나, 생각없이 짠 단순 맹한 클래스 & 함수  

# notebook에 직접 그래프를 plot  
get_ipython().magic('matplotlib inline')


# In[2]:

import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)


# ### 실행 환경
# - Python 3.6.0
# - Anaconda 4.3.0
# - 추가로 seaborn : "conda install seaborn"
# - 추가로 colormap & easydev : "pip install colormap easydev"

# In[3]:

np.__version__ , pd.__version__, seaborn.__version__


# In[4]:

# package_list = ['pandas', 'numpy', 'IPython', 'seaborn', 'sklearn', 'matplotlib', 'statsmodels']
# for pack in package_list:
#     statement = 'import ' + pack
#     exec(statement)
#     print ("%s : %s" % (pack, eval(pack).__version__) ) 


# ## Data Load
# 
# - 책에서 사용한 **Advertising** 데이터를 load 함. 
# - local 머신에서 로딩할 수도, 또는 웹에서 직접 갖고 올 수도 있다. 로딩하기 전에 데이터 구조를 잘 살핍시다  

# In[5]:

# 웹에서 직접 pandas의 DataFrame으로 읽음.  첫째 column을 row index로 사용. 
# 아래의 웹에서 가져 온 csv 파일의 column명이 소문자로 시작하여 에러를 일으킴.  주의...
# advertising = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# or, you can read data as DataFrame from local file system.  
advertising = pd.read_csv('../Data/Advertising.csv', usecols=[1,2,3,4])   
advertising.head()    # advertising.tail() 


# In[6]:

type(advertising)


# In[7]:

advertising.shape


# - 200 개의 row (레코드, observation, sample)이 있음. Column은 4 개  

# In[8]:

advertising.index , advertising.columns    # row index, column names


# In[9]:

advertising.info()        


# 데이터에 대한 자세한 정보 제공 : 타입, shape, 각 feature/column의 속성   
# - **자주 사용하기 바람**

# In[10]:

st(advertising)       # R의 str() 같이 동작하도록 만든 간단한 함수  


# 
# 
# ### 간단한 Exploratory Analysis: 모델링을 하기 전에 데이터의 특성을 살펴본다
# 

# In[11]:

# seaborne 패키지를 이용해 feature들의 scatter plot을 본다  
seaborn.pairplot(advertising)


# In[12]:

# 'Sales'와 feature들간의 관계만을 scatterplot으로 나타내고, 
# R의 ggplot에서와 같이 regression line과 95% 신뢰대역을 나타내도록 함 ('kind='reg').
seaborn.pairplot(data=advertising, x_vars=['TV', 'Radio', 'Newspaper'], y_vars=['Sales'], size=6, aspect=0.8, kind='reg')


# In[13]:

Corrplot(advertising).plot(fontsize='large')    # R style Corrplot 
plt.show()


# - Sales와 TV간 ***강한 정비례*** 관계가 있다 

# ## 3.1 Simple Linear Regression : *feature가 1개*
# 
# 
# $Y = \beta_0 + \beta_1X$
# 
# - $Y$ : response/output/target 
# - $X$ : feature/input/predictor
# - $\beta_0$ is the intercept
# - $\beta_1$ is the coefficient for $X$
# 
# Response(Y)로 *sales*, 1개의 feature(X)를 *TV*로 삼으면, 
# 
# $sales = \beta_0 + \beta_1TV$
# 
# - $\beta_0$ 와 $\beta_1$ 들을 **model coefficients (또는 weight)** 라 함 
# - **simple linear regression의 학습** : **sales**와 **TV** 관계에 가장 맞는(RSS를 최소화하는) 선형식을 구성하는 $\beta_0$와 $\beta_1$을 데이터를 보고 학습해 추정한다

# ## Estimating the Coefficients of Linear Model 
# ***Statsmodels*** 을 사용해 **advertising** 데이터에 대한 linear regression 모델의 coefficient 추정
# - ### [statsmodels version 0.5](http://statsmodels.sourceforge.net/stable/example_formulas.html) 부터 R 스타일 formula 형태 추가   

# ## Statsmodels의 Linear Model 사용하기
# 1. **모델 import** : 우리는 위에서 이미 "import statsmodels.formula.api as smf" 하여 관련 모듈(api)를 'smf' 라는 alias로 가져옴
# 2. **모델 instantiate** : 클래스 생성자를 이용해 모델을 만듬. 이 때 argument로 regression formula 포함
# 3. **학습 시킴** : instantiate된 모델 객체에게 fit() 명령을 내려 학습/훈련시키고, 학습된 모델을 반환 받음
# 4. **학습된 모델 활용** : 학습된 모델을 이용해 새로운 입력에 대해 예측을 하던가 등, 적절한 일거리를 줌
# 

# In[14]:

# 1. 모델 import : 모델을 포함하는 모듈을 이미 import 했음  

# 2. Model Instantiation: Ordinary Least Squares (ols) 방식 linear regression 모델 만들기
#    - 입력 데이터는 DataFrame 타입 

lm = smf.ols(formula='Sales ~ TV', data=advertising)   

# 'advertising' DataFrame에서 'Sales' column을 response로, 'TV' column을 feature로 하는
#  linear regression 모델을 정의함 

# 3. 모델에게 학습 시키고, 그 결과인 (학습된) 모델을 'lm_learned'으로 받음 
lm_learned = lm.fit()

# 학습된 모델의 coefficients
lm_learned.params

# lm_learned.pvalues            # p values
# lm_learned.rsquared           # R-squared statistic 


# - **lm_learned._Tab_를 쳐서 'lm_learned' 객체에 어떤 method를 쓸 수 있는 지 보도록**
#   

# In[15]:

# 보통은 위 2 & 3번 과정을 연결(chaining)함  
lm = smf.ols(formula='Sales ~ TV', data=advertising).fit()   
# 학습한 모델 (즉, fit model)이 만들어졌음

print ("Coeffients:\n%s \n\np-values:\n%s , \n\nr-squared: %s " % (lm.params, lm.pvalues, lm.rsquared))


# ### 다음 두 개의 cell은  response와 feature간의 관계를 시각화하는 또 다른 예 

# In[16]:

# Sales를 Y-축에, TV 광고비를 X-축에 놓은 scatter plot을 그리자   
plt.scatter(advertising.TV, advertising.Sales)
plt.xlabel("TV (in 1000's)")
plt.ylabel("Sales (in 1000's)")

# 위 plot에 simple regression 선을 overlay 
X = pd.DataFrame({'TV':[advertising.TV.min(), advertising.TV.max()]})
Y_pred = lm.predict(X)
plt.plot(X, Y_pred, c='red')
plt.title("Simple Linar Regression")


# In[17]:

# seaborn 패키지를 이용할 수도 
seaborn.regplot(advertising.TV, advertising.Sales, order=1, ci=None, scatter_kws={'color':'r'})
plt.xlim(-50,350)
plt.ylim(ymin=0)
plt.grid()


# In[18]:

lm.summary()     #  모델 전체 요약. R의 summary() 함수와 비슷 


# In[19]:

# ISLR - Table 3.1
lm.summary().tables[1]


# In[20]:

st(advertising)


# ## 학습된 모델 활용: 예측과 관련 이슈 들 
# - 위에서 만든 모델 lm은 Advertising의 TV 변수만을 feature로 사용해 만들었음
# - 위의 R-squared 값 0.612 이나 Residual은 모델을 만들 때 사용한 데이터 (Training set)를 이용해 구한 Training Performance  
# - 예측분석의 목표는 training set에 대해 좋은 성능을 보이는 모델을 만듬이 아니라, 처음 보게 될 (미래)의 out-of-sample 데이터에 대해 좋은 성능을 보일 것 같은 모델을 만드는 것 (즉, generalize 잘 하여 out-of-sample 성능이 좋은 모델)
# - 미래의 데이터가 지금 존재하지 않는데 현재의 모델이 미래에 어떻게 동작할 지 짐작할 수 있을까? -> **모델 평가 **
# 
# 
# #### 예측 : 만들어진 모델 (lm)을 이용해 새로운 predictor 값 (TV)을 줄 때 'Sales' 예측은? 
# - 가령, TV = 100 일 때 Sales 예측

# In[21]:

# statsmodel formula 인터페이스는 입력을 pandas의 DataFrame 같은 array 형태 데이터 구조로 주어야 함 
x_new = pd.DataFrame({'TV': [100]})    # dictionary로 df를 만드는 일반 방법 
# x_new.info()
x_new.head()


# ### 4. 예측 : 아래에서와 같이 'predict' 메소드를 이용 
# - ** predict() 의 입력이 DataFrame 같이 array 형태로 training에 사용했던 feature들을 갖고 있어야 함**

# In[22]:

lm.predict(x_new)    # 결과인 예측치를 numpy의 ndarray로 반환 


# ### 손으로 계산하여 확인하면; 
# $$y = \beta_0 + \beta_1x$$
# $$y = 7.0326 + 0.0475 \times x$$

# In[23]:

sales_manual = lm.params.Intercept + lm.params.TV * 100
print("Manual Calculation : %6f" % sales_manual)


# In[24]:

X_new = pd.DataFrame({'TV': [100, 422, 74]})   # TV가 100, 422, 또는 74일때 Sales 예측은? 
lm.predict(X_new)


# 
# 
# 
# # Multiple Linear Regression
# 
# **multiple linear regression**: 여러 feature들을 사용해 response 추정 
# 
# $Y = \beta_0 + \beta_1X_1 + ... + \beta_nX_n$
# 
# ***Advertising***의 TV, Radio, Newspaper들을 feature로 하고, Sales를 response로 한 multiple linear regression :
# 
# $Sales = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper$
# 

# In[25]:

lm_mul = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=advertising).fit()
lm_mul.summary()


# ### 결과 해석
# - F statistic의 p-value가 매우 작으므로 (1.58e-96)  모델은 유효 (최소한 1개 이상의 variable이 response와 관련)
# - TV와 Radio의 p-value는 의미있음.  하지만 Newspaper의 p-value는 0.86에 달하므로 "Newspaper가 response와 관련이 없다"라는 null-hypothesis를 거부할 수 없음. 따라서 Newspaper 변수를 모델에 포함하기에는 적합하지 않음 
# - **R-squared**가 0.89로 simple linear regression (0.612) 때보다 증가. 이 모델이 최소한 simple linear regression 보다 traning set의 response를 더 잘 설명(예측)한다고 생각할 수 있음. 
# - 주의: 이 R-squared는 모델을 만들 때 데이터 (즉, training set에)에 대해서 구한 것이기에 실제 환경에서도 (out-of-sample) 더 좋은 특성을 보이는 지는 확신할 수 없음 
# - **Cross-validation**와 같은 평가 방법을 통해 모델이 **out-of-sample**에 대해서도 **generalize** 잘 할까 짐작해 볼 수 있음 --> 나중에 

# In[26]:

lm_mul.summary().tables[1]               # Table 3.4 of ISLR 


# In[27]:

advertising.corr()     # Table 3.5 of ISLR : correlation matrix (상관 관계)


# ## 3.3   Other Considerations in the Regression Model
# 
# ### Qualitative Predictors

# In[28]:

# Load 'credit' data from local file system 
credit = pd.read_csv('../Data/Credit.csv', usecols=list(range(1,12)))
credit.info()


# - 위 feature들의 data type (dtypes)에서 float64, int64와 같이 숫자가 아닌 'object' 인 것들은 대부분 string 또는 다른 클래스 타입. 이것들이 category 타입 변수일 가능성 많음.
# - Feature중 Gender, Student, Married, Ethnicity 변수가 qualitative(categorical) 변수
# - 400개의 row/observation이 있는데, 모든 feature들이 400 개의 non-null 값을 지님. 즉, missing value가 없음

# In[29]:

credit.head(3)


# In[30]:

credit.isnull().sum()           # 다시 missing value 없음을 확인 


# In[31]:

seaborn.pairplot(credit[['Balance','Age','Cards','Education','Income','Limit','Rating']])  # ISLR - Fig 3.6
# 실행 시간이 조금 걸림.  Wait.


# In[32]:

Corrplot(credit[['Balance','Age','Cards','Education','Income','Limit','Rating']]).plot(fontsize='large')     
plt.show()


# Interpreting the ***corrplot***
# - 파란색(붉은색)으로 갈수록 Positive(Negative) Correlation
# - 긹죽한 타원형태가 될수록 correlation이 강함   

# In[33]:

credit.Gender.unique()               # Gender 변수는 단 2개의 category를 갖음    


# ## 카테고리형 변수 'Gender'를 feature로 활용

# In[34]:

lm_cat = smf.ols(formula='Balance ~ Gender', data=credit).fit()   # Gender has 2 levels -> 1 dummy variable
lm_cat.summary().tables[1]          # ISLR - Table 3.7  


# In[35]:

# Regression of Balance onto Ethnicity
lm_cat_Eth = smf.ols('Balance ~ Ethnicity', credit).fit()
lm_cat_Eth.summary()            # Table 3.8 


# - F-statistic p-value가 0.957에 달해 'Balance와 Ethnicity간 관련이 없다'는 null hypothesis를 거부할 수 없기에 이 데이터에 따르면 null hypothesis를 따른는 것이 좋다.  즉, 이 모델은  **꽝!**

# In[36]:

st(credit)


# **변수들 중 'Ethnicity'만 제외하려면 - formula에 feature 다 나열하기 귀찮음. 뒤에... **

# In[37]:

lm_all = smf.ols('Balance ~ Income + Limit + Rating + Cards + Age + Education + Gender + Student + Married', credit).fit()
lm_all.summary()


# 
# ## Removing the Additive Assumptions : 변수간 Interaction 

# In[38]:

# TV와 Radio간 interaction term을 주고 linear model을 만들면
lm_interact = smf.ols('Sales ~ TV + Radio + TV:Radio', advertising).fit()
lm_interact.summary().tables[1]             # Table 3.9


# - TV와 Radio간 interaction이 유효
# 
# 

# In[39]:

smf.ols('Sales ~ TV*Radio', advertising).fit().summary().tables[1]      # 앞의 formula를 이렇게 표현 가능  


# In[40]:

smf.ols('Sales ~ TV + Newspaper*Radio', advertising).fit().summary()


# - Newspaper와 Radio간 interaction은 유효하지 않음  

# 
# ### Interaction between qualitative variable and a quantitative variable

# In[41]:

# Income(quantitative) 과 Student(qualitative with 2 levels)간 Interaction이 없다하고 모델을 학습하면;
lm_no_interact = smf.ols('Balance ~ Income  + Student', credit).fit()   
lm_no_interact.summary()


# In[42]:

# Income(quantitative) 과 Studen(qualitative with 2 levels)간 Interaction이 있게 만들면;
lm_interact = smf.ols('Balance ~ Income*Student', credit).fit()
lm_interact.summary()


# - 'Income'과 'Student' 사이의 interaction이 없다고 생각하는 것이 옳으며, 이는 R-square 값이 거의 증가하지 않은 것을 통해서도 짐작할 수 있다.

# 
# ### Non-linear relationships using polynomial regressions

# In[43]:

# load 'Auto' data
auto = pd.read_csv('../Data/Auto.csv')
auto.info()
auto.head()


# **(중요) horsepower 변수가 숫자이어야 함. 그런데, 위의 auto.info()로 본 horsepower 변수 타입이 'object'로 되어 있음.  즉 숫자가 아니라고 함.  auto.head()로 보니 처음에는 분명 숫자.  따라서 horsepower 변수 중간 어디 즈음 숫자가 아닌 것이 있음 **

# In[44]:

# Find out which rows have non-numeric value on 'horsepower' column
auto_problem = auto[auto.horsepower.apply(lambda x: not(x.isnumeric()))]
auto_problem


# 5개의 observation 들이 'horsepower' feature에 숫자가 아님.  원본 auto.csv 를 보고 확인  
# - 위의 row들을 제거할 수도 있고, 또는 파일을 읽을 때 위의 문제가 있는 row들을 제거하고 읽을 수도 있음 

# In[45]:

# Read the data again. This time skipping problematic rows 
auto = pd.read_csv('../Data/Auto.csv', na_values='?').dropna()
auto.info()
auto.iloc[28: 34, :]


# - 문제있는 row들이 제거됨을 확인

# 
# ### mpg를 $horsepower$ 와  $horsepower^2$ 에 대해 regression 

# In[46]:

# OLS regression of mpg onto horsepower and squared(horsepower)
lm_quadratic = smf.ols('mpg ~ horsepower + np.square(horsepower)', data=auto).fit()
lm_quadratic.summary().tables[1]             # ISLR - Table 3.10


# In[47]:

# Polynomial regression upto 3'rd degree 
lm_deg3 = smf.ols('mpg ~ horsepower + np.power(horsepower,2) +  np.power(horsepower,3)', data=auto).fit()
lm_deg3.summary()


# - R의 poly()같은 함수 만드는 것은 쉬움. 함수로 만들 가치 없음 .

# # 3.6 Lab: Linear Regression

# ##  sanity check

# In[48]:

Boston = pd.read_table("../Data/Boston.csv", sep=',')


# In[49]:

st(Boston)
Boston.head()


# - 모든 column들이 숫자(numeric) 임

# In[50]:

Boston.describe()


# In[51]:

Boston.isnull().sum()


# In[52]:

Boston.columns        # pandas DataFrame 클래스는 'columns' attribute을 갖고 있음 


# ## 3.6.2 medv를 response, lstat를 predictor로 한 simple regression

# In[53]:

lm_fit = smf.ols(formula='medv ~ lstat', data=Boston).fit()


# In[54]:

lm_fit.summary()


# In[55]:

lm_fit.resid.describe()      # Residuals statistics


# *** 신뢰구간 ***

# In[56]:

lm_fit.conf_int(alpha=0.05)      # default alpha=0.05 : 95% confidence interval


# **[참고](http://statsmodels.sourceforge.net/devel/examples/generated/example_ols.html) : OLS Prediction with confidence interval ** 

# In[57]:

from statsmodels.sandbox.regression.predstd import wls_prediction_std

X_new = pd.DataFrame({'lstat':[5,10,15]})
lm_fit.predict(X_new)


# In[58]:

plt.scatter(Boston.lstat, Boston.medv )

X = pd.DataFrame({'lstat':[Boston.lstat.min(), Boston.lstat.max()]})
Y_pred = lm_fit.predict(X)
plt.plot(X, Y_pred, c='red')
plt.xlabel("lstat")
plt.ylabel("medv")


# # 3.6.3 Multiple Linear Regression

# In[59]:

lm_fit = smf.ols('medv ~ lstat+age', data=Boston).fit()
lm_fit.summary()


# ### R의 "formula = medv ~ ." 같이 medv를 제외한 다른 모든 column을 predictor로 삼는 간편 식이 python에 없음.  그냥 다음과 같이 하면 됨.

# In[60]:

# Response인 'medv'를 제외한 모든 column들을 feature로 삼으려면,
columns_selected = "+".join(Boston.columns.difference(["medv"]))
my_formula = "medv ~ " + columns_selected
my_formula


# * 단순 조작이기에 함수로 만들 필요 없겠죠...  참고로, formula에서 R 처럼 '-'도 먹힘  

# In[61]:

lm_fit = smf.ols(formula = my_formula, data=Boston).fit()


# In[62]:

lm_fit.summary()


# In[63]:

lm_fit.resid.describe()       # Residuals statistics


# In[64]:

# 'age' 를 제외한 다른 모든 변수들을 predictor로 삼으려면
columns_selected = "+".join(Boston.columns.difference(["medv", "age"]))
my_formula = "medv ~ " + columns_selected
lm_fit1 = smf.ols(formula = my_formula, data=Boston).fit()
lm_fit1.summary().tables[1]


# In[65]:

lm_fit1.resid.describe()


# ## 3.6.4 Interaction Terms

# In[66]:

lm_fit = smf.ols('medv ~ lstat*age', data=Boston).fit()
lm_fit.summary()


# In[67]:

Boston.head()


# ### 임의의 test set을 만들어 response를 예측해 봄 

# In[68]:

# Interaction term이 있지만 이는 'age'와 'lstat' 변수에서 파생된 것이기에 이 두 변수만 필요함 
test = pd.DataFrame({'age':[65.4, 79, 23], 'lstat':[4.8, 10, 5]})
test


# In[69]:

lm_fit.predict(exog=test)


# - "predict()의 'exog' 같은 단어들은 어디서 유래했을까" 가 궁금하면 [여기로](http://statsmodels.sourceforge.net/devel/endog_exog.html) 

# In[70]:

# residual 을 계산해 봄. 모델을 fit할 때 사용하지 않은 변수인 'rm'을 예측 변수로 넣어도 에러 발생 않함 
y_predict = lm_fit.predict(Boston.loc[:,['age', 'lstat',  'rm']])   # training set에 대한 prediction
(Boston.medv - y_predict)[0:5]     


# In[71]:

lm_fit.resid[:5]            # 위의 결과와 같음 


# In[72]:

# lm_fit.predict(Boston.loc[:,['age', 'rm']])    
# 'lstat'이 없다고 exception 일으킴 


# ## 3.6.5 Non-linear Transformation of the Predictors

# In[73]:

lm_fit2 = smf.ols('medv ~ lstat + np.power(lstat, 2)', data=Boston).fit()
lm_fit2.summary()


# ### ANOVA test to compare two models. [(참고)](http://statsmodels.sourceforge.net/devel/generated/statsmodels.stats.anova.anova_lm.html) 

# In[74]:

import statsmodels.api as sm

lm_fit = smf.ols('medv ~ lstat', data=Boston).fit()
table = sm.stats.anova_lm(lm_fit, lm_fit2, typ=1)
print(table)


# ## 3.6.6 Qualitative Predictors

# In[75]:

Carseats = pd.read_csv("../Data/Carseats.csv", index_col=0)
Carseats.head()


# In[76]:

Carseats.columns


# In[77]:

Carseats.info()


# In[78]:

columns_selected = "+".join(Carseats.columns.difference(["Sales"]))
my_formula = "Sales ~ Income:Advertising + Price:Age + " + columns_selected  
my_formula


# In[79]:

lm_fit = smf.ols(my_formula, data=Carseats).fit()
lm_fit.summary()


# In[80]:

Carseats.head()


# In[81]:

Carseats_training = Carseats.loc[:,'CompPrice':]
# Carseats_training


# In[82]:

lm_fit.predict(Carseats_training)[:5]       # training set feature를 이용해 training set response 추정  


# In[83]:

(Carseats.Sales - lm_fit.predict(Carseats_training)).describe()    # residual statistics w.r.t. training set


# -----------------------------------------------------------------------------

# # [scikit-learn](http://scikit-learn.org/stable/) 
# - ### Python을 위한 High-Level 머신러닝 libaray
# - ### 체계적 구조(사용법)에 따른 편의성/재사용성, 짜증 감소에 따른 스트레스 저하 
# - ### 예술적 수준의 [Documentation](http://scikit-learn.org/stable/user_guide.html)
# - ### 확장성, 유연성, 합리성, 유머, 전반적 높은 수준 SW, 우수한 자동화 및 Production System화, Contribution하고 싶은 마음을 일으키며  
# - ### 다른 언어나 다른 Python 머신 러닝 library에 비교해 불편한 점도 있으나 위와 같은 장점이 돋보임 
# 

# ## Scikit-learn이 지원하는 머신러닝 알고리즘 및 적합한 알고리즘 선택 요령
# <img src="http://scikit-learn.org/stable/_static/ml_map.png">

# -----------------------------------------------------------------

# # scikit-learn [Linear Regression](http://scikit-learn.org/stable/modules/linear_model.html)

# ### 말 그대로 가장 밋밋한 Linear Regression 알고리즘인 [Ordinary Least Squares Linear Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) 연습 - 3장의 Linear Regression 내용은 이것으로 다 됨 

# ## 간략 scikit-learn [소개](http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting)
# - 학습 모델(알고리즘)을 **estimator** 라 함. 정확한 정의는: an estimator is a Python object that implements the methods fit(X, y) and predict(T) 
# 
# ### scikit-learn의 estimator들은 데이터에 대해 다음과 같은 조건을 요구함
# 1. feature와 response가 각각 독립적인 객체
# 2. feature와 response가 숫자
# 3. feature와 response가 NumPy ndarray이거나 또는 DataFrame, Python array같이 쉽게 ndarray로 변환 가능 해야 함. 또한, scipy의 sparse matrix를 허용.
# 4. feature는 2-D ndarray로 shape가 (n_samples, n_features) 이어야 함. Response는 'n_samples' 길이의 vector 이어야 함.  
# 
# ## scikit-learn의 estimator 사용 패턴  
# ### 1. estimator를 import 
#   - ***from  sklearn.linear_model  import  LinearRegression***
#   <br> 
#   
# ### 2. instantiate the estimator ###
#   - ***model = LinearRegression()*** : instantiate할 때 estimator에서 hyperparameter를 지정해 튜닝
#     <br>
#     
# ### 3. estimator에게 데이터(X:feature, y:response)를 제공해 학습시킴 (모델 traing/fit  ###
#   - ***model.fit(X, y) ***    : fit가 in-place 진행, 즉 결과가 model 내부에 저장됨 
#     <br>
#     
# ### 4. 학습된 estimator에 새로운 데이터의 feature(X_new)를 주고 response를 추정 ### 
#   - ***y_predict = model.predict(X_new) *** 
# 

# ## Advertising 데이터를 이용

# ### 1. Estimator (여기서는 'LinearRegression')을 갖고 옴 

# In[84]:

from sklearn.linear_model import LinearRegression   # sklearn : scikit-learn 을 말함 
# from sklearn import datasets         


# ### 'LinearRegression' estimator가 쓸 수 있도록 data 구조 만들기 

# In[85]:

advertising.head()


# In[86]:

advertising.info()


# - Sales를 response, 나머지 TV, Radio, Newspaper를 feature 삼으려 함
# - response와 feature들이 모두 숫자 --> scikit-learn의 data 조건-2 만족

# In[87]:

# X 와 y  각각 만들기 
X = advertising.loc[ :, ['TV', 'Radio', 'Newspaper'] ]   # DataFrame 타입 
y = advertising.Sales
print(X.head(), '\n')
print(X.values[:5], '\n')
print(type(X.values))


# - pandas의 DataFrame 객체는 데이터를 numpy.ndarray 형태로 내부에 갖고 있다

# In[88]:

X.shape , X.values.shape         # DataFrame X의 모양, 내부 ndarray의 모양이 같음  


# In[89]:

type(y)


# In[90]:

y.head()


# In[91]:

y.values


# In[92]:

type(y.values)


# In[93]:

# y.values.head()     # error.  이유는 head()는 pandas DataFrame, Series 메소드. numpy.ndarray에 안됨 
y.values[:5]


# In[94]:

y.shape


# In[95]:

y.values.shape       # pandas and numpy classes both support shape() method 


# ### DataFrame과 Series는 데이터를 내부에서 numpy.ndarray로 관리.  
# - *** scikit-learn estimator들은 DataFrame과 Series 데이터 구조도 받아드린다***

# ### 2. Estimator를 instantiate 

# In[96]:

model = LinearRegression()


# ### 3. Estimator를 훈련  

# In[97]:

model.fit(X, y)


# ### 학습된 Estimator 살펴보기 :
# - model.[Tab] 을 하여 어떤 메소드가 있는 지 보자

# In[98]:

print(model.coef_)            # feature matrix 'X'의 feature 순서대로, 즉 TV', 'Radio', 'Newspaper'
list(zip(X.columns, model.coef_ ))


# - 앞에 statsmodels linear model의 결과와 같음 

# In[99]:

model.intercept_


# In[100]:

model.residues_             # Residual sum of squares (RSS).  0.18에 생겼는데 0.19에 deprecate   


# - scikit-learn에서 coef\_, intercept\_  같이 estimator attribute명 뒤에 '_'가 붙은 것은 (학습된) 모델의 attribute임을 나타냄. 따라서 학습되지 않은 estimator에 위 멤버를 요청하면 에러    

# In[101]:

model.score(X, y, sample_weight=None)         # R-squared 


# ### 4. Predict (예측/추정) : response 추정 
# - 앞 단계에서 estimator가 훈련을 통해 학습이 됨
# - 이 estimator로 feature가 입력될 때 response를 추정해 본다 

# In[102]:

# training 할 때 사용한 X를 그대로 feature로 삼아 response를 보자
y_pred = model.predict(X)
pd.DataFrame({'y_True': y, "y_pred": y_pred}).head(10)     


# In[103]:

# RSS manual 계산과 비교  
np.square(y - y_pred).sum(), model.residues_


# In[104]:

X.tail()


# In[105]:

X.values[-5:]


# ### 기본적으로 predict 메소드의 입력 feature는 numpy ndarray이어야 함

# In[106]:

X_new = np.array([[45.4, 12, 44]])     # One observation with features TV, Radio, Newspaper order 
X_new.shape


# - X_new가 2D ndarray 이어야 함.  X_new는 1x3 array.  column 배열이 estimator를 훈련시킬 때의 X column 순서와 같아야 함  

# ### 새로운 feature에 대한 response 추정 

# In[107]:

model.predict(X_new)


# ### predict()는 입력 feature로 DataFrame과 Python array도 잘 받아드린다.  단 2D 이어야 함

# In[108]:

X_new = pd.DataFrame([[45.4, 12, 44]])
model.predict(X_new)              # OK


# In[109]:

X_new = [[45.4, 12, 44]]
model.predict(X_new)             # OK


# In[110]:

X_new


# In[111]:

X_new = [45.4, 12, 44]
model.predict(X_new)         # 아직은 됨.  곧 에러로 취급한다고.  2D array로 만드라는 말 


# In[112]:

X_new = pd.Series([45.4, 12, 44])      # 위와 같은 주의    
model.predict(X_new)


# ### 추천 ###
# #### predict()에 feature array를 줄 때 가능한 DataFrame으로 주자. DataFrame은 2D 데이터 구조이고, 보통 column name도 함께 쓰기에 에러도 준다.  
# 그러나 주의할 점도 있음

# In[113]:

X_new = pd.DataFrame({'TV':[34,44,56], 'Radio':[123,55,23], 'Newspaper':[23,40,121]})
X_new


# In[114]:

# 내부 ndarray를 보면,
X_new.values


# #### X_new의 column 순서가 처음  X_new를 dictionary로 만들 때의 순서와 다름.   이는 Python Dictionary가 순서 개념이 없기 때문임.  원래 순서 TV, Radio, Newspaper 순으로 순서를 맞추어 predict() 메소드에 주어야 함

# In[115]:

X.columns


# In[116]:

X_new = X_new[X.columns]          # X_new의 column 순서를 X 순서에 따라 재배열 함      
X_new.values


# In[117]:

model.predict(X_new)


# #### [Lesson] predict()에게 주는 데이터를 만드는 등, 중요한 ndarray를 만들 경우 녀석이 내가 생각했던 그대로 되어있나 확인하자

# ***Note***
# 
# [scikit-learn OLS API reference](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score) 에 있듯이 이 estimator는 **F-statitic, p-value, confidence interval** 등을 기본 제공하지 않음. 또한, statsmodels과 같이 categorical 변수, interaction, 변수를 non-linear (polynomial) 변환 적용 등이 공짜가 아님. 
# 
# 물론, 이런 모든 것들을 변수들을 미리 preprocessing 하여 가능함. statsmodels이나 R의 모델들은 categorical feature가 있을 시 자동적으로 preprocessing 해 준 것임.  
# 
# scikit-learn과 Python 생태계는 매우 다양하고 강력한 preprocessing library/기능을 제공함.  
# 
# 이 estimator를 파생/확장하여 **categoric 변수지원, interaction 지원, F-statitic, p-value, confidence interval** 등을 제공하는 것은 쉬우나 이는 Python 이나 SW 철학에 어긋남 **(공부하여, 이해하고, 인정하고, 고마와하며 갖다 쓰면 되지, 같은 것을 다시 만들 이유는 없음). 필요하면 statsmodels 쓰면 됨.**
# 
# ISLR 6장에는 여기에서 배운 것의 심화 내용이 있음. 모두 어렵지 않은 내용임. 6장의 내용도 scikit-learn이 잘 지원함.   
# 
# ** 단순하거나, 복잡하거나, preprocessing을 어떻게 했던지 기본적으로 모든 linear 모델은 response와 feature들간에 linear한 관계가 있을 때 잘 동작함 **  
#     

# ## 동영상 참고 
# 
# - [Hastie & Tibshirani의 Ch.3 강의](https://www.youtube.com/watch?v=PsE9UqoWtS4&list=PL5-da3qGB5IBSSCPANhTgrw82ws7w_or9) : ISLR 저자들의 강의  
# - Kevin Markham의 [scikit-learn 강의](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A) : 쉽게, 친절하게 함
# - [Jake VanderPlas: Machine Learning with Scikit Learn](https://www.youtube.com/watch?v=HC0J_SPm9co) : Linear Regression 에 관한 것은 아니나 scikit-learn 으로 iris classficiation에 관한 tutorial 
# - [Machine Learning with Scikit Learn | SciPy 2015 Tutorial | Andreas Mueller & Kyle Kastner Part I](https://www.youtube.com/watch?v=80fZrVMurPM) : scikit-learn Machine Learning 전반에 관한 tutorial 
# - 위 강의들과 함께 있는 url들을 따라 가보도록.  
# 
# 

# #### 마지막으로,

# In[118]:

Carseats.info()


# In[119]:

all_features = '+'.join(Carseats.columns.difference(['Sales']))
my_formula = "Sales ~ " + all_features + " - Population - Education + ShelveLoc:Advertising + Income:Advertising"
print("formula = ", my_formula)
lm_Carseats = smf.ols(formula = my_formula, data=Carseats).fit() 
lm_Carseats.summary() 


# In[120]:

print("exit with 0")


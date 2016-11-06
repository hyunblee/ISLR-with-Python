***Found out [Jordi Warmenhoven](https://github.com/JWarmenhoven/ISLR-python.git) has supplied us with a great python version of ISLR. 
It's good to know that there's a place that I could learn python technique on this wonderful subject***  

# ISLR-with-Python
### Introduction to Statistical Learning in R (ISLR)을 Python으로  
-  강의 슬라이드나 관련 자료가 함께 있음 

  
### 3장 - Linear Regression
* statsmodel 패키지 사용하여  
* scikit-learn의 OLS estimator 사용하여


### 4장 - Classification 
* Logistic Regression : scikit-learn estimator와 statsmodels 라이브러리 사용하여, 
* KNN Regression과 Classification : scikit-learn estimator 사용하여,
* Regressor 평가 (Evaluation Metric) : MAE, MSE, RMSE 
* Classifier 평가 : Confusion Matrix, ROC, AUC
* Train/Test Split 방법, Cross-Validated AUC 짧은 소개 

### 5장 - Resampling Methods : Model Evaluation
* Validation Set Approach (Train/Test Split)
* K-Fold Cross Validation
  - Regression Model의 Test MSE 추정
  - Hyper Parameter 튜닝 : KNN Regression 에서 K 선정 
  - Classification Model에서 Cross-Validated AUC 계산
  
### Preprocessing for scikit-learn
* scikit-learn의 LinearRegression estimator에 적용하기 위해  
* categorical 변수, polynomial regression, interaction preprocessing 
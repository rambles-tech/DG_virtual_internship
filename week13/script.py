import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.utils import resample

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


rand_state = 25

# bring in cleaned dataset
df = pd.read_csv('data/data.csv', index_col='ptid')

# fix column names for XGBoost
df.rename(columns={'age_bucket_<55':'age_bucket_under_55', 'age_bucket_>75':'age_bucket_over_75'}, inplace=True)

# Slice the persistent and non-persistent groups
non_persistency = df[df["persistency_flag"] == 0]
persistency     = df[df["persistency_flag"] == 1]

# select the appropriate number of non-persistent subjects
select_non_persistent = resample(non_persistency,
                           replace=False,
                           n_samples=len(persistency),
                           random_state=rand_state)

balanced_df = pd.concat([persistency, select_non_persistent])

def prep_data(data):
    X = data.drop("persistency_flag", axis=1)
    y = data["persistency_flag"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rand_state)

    return (X, y, X_train, X_test, y_train, y_test)


X, y, train_features, test_features, train_labels, test_labels = prep_data(balanced_df)


# Importing packages and settings: 
import warnings 
warnings.filterwarnings(action= 'ignore')
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import joblib


# Creating an object with the column labels of only the categorical features and one with only the numeric features:
categorical_features = X.select_dtypes(exclude="number").columns.tolist()
numeric_features = X.select_dtypes(include="number").columns.tolist()
# Create the categorical pipeline, for the categorical variables Aki imputes the missing values with a constant value and we encode them with One-Hot encoding:
categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy= "constant", fill_value= "unknown")), 
        ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)) 
    ]
)
# Create the numeric pipeline, for the numeric variables Aki imputes the missings with the mean of the column and standardize them, so that the features have a mean of 0 and a variance of 1: 
numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), 
           ("scale", StandardScaler())]
)
# Combining the two pipelines with a column transformer:
full_processor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# Instantiate the XGBClassifier:
xgb_cl = xgb.XGBClassifier(eval_metric="logloss", seed=rand_state) 
# Create XGBoost pipeline:
xgb_pipeline = Pipeline(steps=[
    ("preprocess", full_processor),
    ("model", xgb_cl)
])

# Evaluate the model with the use of cv:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand_state)  #, shuffle=True with or without shuffle??
scores = cross_val_score(xgb_pipeline, train_features, train_labels, cv=cv, scoring = "roc_auc")
print("roc_auc = %f (%f)" % (scores.mean(), scores.std()))


def print_results_gridsearch(gridsearch, list_param1, list_param2, name_param1, name_param2):
  
    # Checking the results from each run in the gridsearch: 
    means = gridsearch.cv_results_["mean_test_score"]
    # stds = gridsearch.cv_results_['std_test_score']
    # params = gridsearch.cv_results_['params']

    #Visualizing the results from each run in the gridsearch: 
    scores = np.array(means).reshape(len(list_param1), len(list_param2))
    for i, value in enumerate(list_param1):
        plt.plot(list_param2, scores[i], label= str(name_param1) + ': ' + str(value))

    plt.legend()
    plt.xlabel(str(name_param2))
    plt.ylabel('ROC AUC')  
    plt.show()

    # Checking the best performing model:
    print("Best model: roc_auc = %f using %s" % (gridsearch.best_score_, gridsearch.best_params_))



def hyper_param_tuning(model__learning_rate = [None], 
                    model__n_estimators = [None], 
                    model__max_depth = [None], 
                    model__min_child_weight = [None], 
                    model__subsample = [None], 
                    model__colsample_bytree = [None],
                    model__gamma = [None],
                    model__reg_lambda = [None]):

    # Defining the parameter grid to be used in GridSearch:
    param_grid = {  "model__learning_rate":     model__learning_rate, 
                    "model__n_estimators":      model__n_estimators,
                    "model__max_depth":         model__max_depth, 
                    "model__min_child_weight":  model__min_child_weight,
                    "model__subsample":         model__subsample, 
                    "model__colsample_bytree":  model__colsample_bytree,
                    "model__gamma":             model__gamma,
                    "model__reg_lambda":        model__reg_lambda
                }
                
    #instantiate the Grid Search:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand_state)
    grid_cv = GridSearchCV(xgb_pipeline
                            , param_grid
                            , n_jobs=-1
                            , cv=cv
                            , scoring="roc_auc") 

    # Fit
    _ = grid_cv.fit(train_features, train_labels)

    return grid_cv, param_grid



print("Evaluating...")

grid_cv, param_grid = hyper_param_tuning(
                    model__learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1],
                    model__n_estimators = range(50,500,50),
                    # SEARCH PARAMS
                    model__gamma = [i/10.0 for i in range(0,4)],
                    model__reg_lambda = [0, 0.5, 1, 1.5],
                    model__subsample = [i/10.0 for i in range(4,10)], 
                    model__colsample_bytree = [i/10.0 for i in range(4,10)],
                    model__max_depth = range(3,10,2), 
                    model__min_child_weight = range(1,6,2)
                   
                   )


# Predict with Aki's final XGBoost with the best parameters resulting from the GridSearch: 
y_pred_aki = grid_cv.predict(X_test)
y_pred_prob_aki = grid_cv.predict_proba(X_test)[::,1]
# Evaluate:
print("roc_auc_score:",metrics.roc_auc_score(y_test, y_pred_aki))


# Fit Meta's default XGBoost pipeline:
xgb_pipeline.fit(X_train, y_train)
# Predict:
y_pred_meta = xgb_pipeline.predict(X_test)
y_pred_prob_meta = xgb_pipeline.predict_proba(X_test)[::,1]
# Evaluate:
print("roc_auc_score:",metrics.roc_auc_score(y_test, y_pred_meta))


#Saving Aki's final XGBoost pipeline:
best_pipe_aki = grid_cv.best_estimator_
joblib.dump(best_pipe_aki, 'best_pipe_aki.joblib')
import numpy as np
import pandas as pd
import joblib

X_train = np.load('matrices/X_train.npy')
X_test = np.load('matrices/X_test.npy')
y_test = np.load('matrices/y_test.npy')

print('Xtrain successfully loaded: ')
print(X_train)

# load the model from disk
loaded_model = joblib.load('models/finalized_model.pkl')

print('model successfully loaded: Score is-')
result = loaded_model.score(X_test, y_test)
print(result)




import lime
import lime.lime_tabular

# Lambda function for getting predicted probability of target variable
predict_fn_rf = lambda x: loaded_model.predict_proba(x).astype(float)

# Lining up feature names
feature_names = ['RiskPerformance', 'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']


# Creating the LIME Explainer

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names = feature_names[1:], 
                                                   class_names = ["High Credit Risk", "Low Credit Risk"], 
                                                   verbose=True,
                                                   categorical_features =['RiskPerformance'],
                                                   categorical_names = ['RiskPerformance'],
                                                   mode='classification',
                                                   discretize_continuous=True,
                                                   discretizer='quartile',
                                                   kernel_width = 3)

# function to generate a LIME explanation for a given observation x_test
def generate_exp1(x_test):
    # Pick the observation for which validation is required
    #pred = loaded_model.predict(x_test)
    exp = explainer.explain_instance(x_test, 
                                    predict_fn_rf, 
                                    num_features = 10)
    #exp.show_in_notebook(show_all = False)
    # print(type(exp))
    listexp = exp.as_list()
    return listexp

# function to parse a LIME explanation list
def parse_explanation_list(exp_list):
    pass


# TESTING OF MODULE
print('X_test: ')
print(X_test[1])
res = generate_exp1(X_test[1]) # for testing purpose
print('Explanation for row 5: ')
print(res)
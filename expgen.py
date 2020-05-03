import numpy as np
import pandas as pd
import joblib
import exptext

X_train = np.load('matrices/X_train.npy')
X_test = np.load('matrices/X_test.npy')
y_test = np.load('matrices/y_test.npy')


# load the model from disk
loaded_model = joblib.load('models/finalized_model.pkl')

# print('model successfully loaded: Score is-')
result = loaded_model.score(X_test, y_test)


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
    pred = loaded_model.predict_proba(x_test.reshape(1, -1)).astype(float)
    pred_good = pred[:,1]
    
    exp = explainer.explain_instance(x_test, 
                                    predict_fn_rf, 
                                    num_features = 10)
    
    
    mapexp = exp.as_map()
    anchs_vec = mapexp[1]

    strexp = exptext.generate_text_explanation(pred_good, x_test, anchs_vec) 
    if pred_good>0.5:
        pred_good = 1
    else:
        pred_good=0
    # pred_good = np.array(np.greater(pred[:,1], 0.5), dtype=int)
    return strexp, pred_good

# TESTING OF MODULE
def test():
    print('**TEST**')
    print(loaded_model.classes_)

    res, pred_good = generate_exp1(X_test[1]) # for testing purpose
    print('Explanation for row 5: ')
    print(res)
    print('predgood: ', pred_good)

    count = 0
    for i in range(0, len(y_test)):
        res, pred_good = generate_exp1(X_test[i]) # for testing purpose
        
        if(pred_good==y_test[i]):
            count = count+1
    
    acc = count/len(y_test)
    print('Accuracy: ', acc)
    return

#test()
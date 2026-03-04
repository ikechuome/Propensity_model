
import pandas as pd
import numpy as np
# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Libraries to import decision tree classifier and different ensemble classifiers

from sklearn.tree import DecisionTreeClassifier
import shap

# Libtune to tune model, get different metric scores
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

import pickle

def make_confusion_matrix(model, y_actual, X_test, labels=[1, 0]):
    '''
    model    : classifier to predict values of X
    y_actual : ground truth
    X_data   : feature set to predict on (defaults to X_test)
    '''
    # Predict on test data — always original, never resampled
    y_predict = model.predict(X_test)
    
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0, 1])
    
    df_cm = pd.DataFrame(
        cm,
        index=['Actual - No', 'Actual - Yes'],
        columns=['Predicted - No', 'Predicted - Yes']
    )
    
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten() / np.sum(cm)]
    
    # Fixed: renamed to annot_labels to avoid overwriting the parameter
    annot_labels = [f"{v1}\n{v2}" for v1, v2 in
                    zip(group_counts, group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(2, 2)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=annot_labels, fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def get_metrics_score(model, X_train_res, y_train_res,X_test,y_test, flag=True):
    '''
    model        : classifier to predict values of X
    X_train_res  : resampled training features (SMOTE applied)
    y_train_res  : resampled training labels (SMOTE applied)
    '''
    score_list = []

    # Training metrics — use RESAMPLED data (what the model was trained on)
    pred_train = model.predict(X_train_res)
    train_acc       = model.score(X_train_res, y_train_res)
    train_recall    = metrics.recall_score(y_train_res, pred_train)
    train_precision = metrics.precision_score(y_train_res, pred_train)

    # Test metrics — use ORIGINAL test data (never resampled, real-world distribution)
    pred_test = model.predict(X_test)
    test_acc       = model.score(X_test, y_test)
    test_recall    = metrics.recall_score(y_test, pred_test)
    test_precision = metrics.precision_score(y_test, pred_test)

    score_list.extend((train_acc, test_acc, train_recall, test_recall,
                       train_precision, test_precision))

    if flag == True:
        print("Accuracy on training set  : ", f"{train_acc * 100:.2f}%")
        print("Accuracy on test set      : ", f"{test_acc * 100:.2f}%")
        print("Recall on training set    : ", f"{train_recall * 100:.2f}%")
        print("Recall on test set        : ", f"{test_recall * 100:.2f}%")
        print("Precision on training set : ", f"{train_precision * 100:.2f}%")
        print("Precision on test set     : ", f"{test_precision * 100:.2f}%")


def run_shap_analysis(model, X_test):
    """Generate and display SHAP values for model interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SHAP values shape:", np.array(shap_values).shape)
    
    # Summary plot
    shap.summary_plot(
    shap_values[:, :, 1],   # all customers, all features, buyer class (index 1)
    X_test,
    plot_type="bar",
    max_display=15
    )
    shap.summary_plot(shap_values[:, :, 1], X_test, max_display=15)
    
def run_shap_analysis_val(model, X_val):
    X_val_features = X_val.drop(columns=['predicted_order', 'order_probability'])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    print("SHAP values shape:", np.array(shap_values).shape)
    
    # Summary plot
    shap.summary_plot(
    shap_values[:, :, 1],   # all customers, all features, buyer class (index 1)
    X_val,
    plot_type="bar",
    max_display=15
    )
    shap.summary_plot(shap_values[:, :, 1], X_val, max_display=15)

# def model_pkl():
#     with open('ik_propensity_model.pkl', 'rb') as f:
#         loaded_model = pickle.load(f)# ✅ indented inside with block
#     return loaded_model
    
def model_test(final_data):
    X_val = final_data.copy()
    with open('../model/ike_propensity_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)  # ✅ indented inside with block
    y_pred = loaded_model.predict(X_val)
    y_proba = loaded_model.predict_proba(X_val)[:, 1]  # probability score
    X_val['predicted_order'] = y_pred
    X_val['order_probability'] = y_proba
    return X_val,y_pred,loaded_model
 
    
def model(final_data):
    y = final_data['ordered']
    X = final_data.drop(columns='ordered')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1,stratify=y)
    # Perform SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    # Fitting the model on RESAMPLED training data
    d_tree = DecisionTreeClassifier(random_state=1)
    d_tree.fit(X_train_res, y_train_res)
    # get_metrics_score(d_tree, X_train_res, y_train_res,X_test,y_test)
    # make_confusion_matrix(d_tree, y_test, X_test)
    # shap_test(d_tree,X_test)
    filename = '../model/ike_propensity_model.pkl'
    pickle.dump(d_tree,open(filename,'wb'))
    return d_tree,X_train_res, y_train_res,X_test,y_test

if __name__ == '__main__':
    #Read data
    path = '../data/final_version'
    final_data = pd.read_csv(path)
    # model(final_data)
    d_tree,X_train_res, y_train_res,X_test,y_test = model(final_data)

    path = '../data/final_test_version.csv'
    final_data = pd.read_csv(path)
    X_val, y_pred, loaded_model = model_test(final_data)
    

# get_metrics_score(d_tree, X_train_res, y_train_res)  # ✅ pass resampled data
# make_confusion_matrix(d_tree, y_test)  # ✅ no change needed here
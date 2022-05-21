from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

import read
import matplotlib.pyplot as plt

def evaluate(y_test, y_pred):
    print("Evaluating ...")
    print("Accuracy is %f." % accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Precision score is %f." % precision_score(y_test, y_pred))
    print("Recall score is %f." % recall_score(y_test, y_pred))
    print("F1 score is %f." % f1_score(y_test, y_pred))
    print("-----------------------------------")

def perceptron(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{
            'penalty': ['l2', 'l1', 'elasticnet'],
            # 'l1_ratio': [0.0001, 0.001, 0.01, 0.15],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'tol': [1e-3, 1e-4, 1e-5]
        }]

        scores = ['f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(Perceptron(), tuned_parameters, cv=5, scoring=score, n_jobs=-1)
            clf.fit(x_train, y_train)

            print("Best parameters set found on development set:")
            print("")
            print(clf.best_estimator_)
            print(clf.best_params_)
            print("")
            print("Grid scores on development set:")
            print("")
            print(clf.best_score_)
            print("")

            print("Detailed classification report:")
            print("")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print("")
            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            print("")
            # pred_proba = clf.predict_proba(x_test)[:, 1]
    else:
        models = [
            Perceptron()
        ]
        for model in models:
            print("Fitting Perceptron ...")
            model.fit(x_train, y_train)
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            # pred_proba = model.predict_proba(x_test)[:, 1]

    # return y_pred_train, y_pred_test, pred_proba
    return y_pred_train, y_pred_test

if __name__ == "__main__":
    print("reading data ...")
    x_all, y_all = read.read()

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
    y_pred_train, y_pred_test = perceptron(x_train, y_train, x_test, y_test, TUNING=True)

    print("Perceptron model's performance on training set:")
    evaluate(y_train, y_pred_train)
    print("")
    print("Perceptron model's performance on testing set:")
    evaluate(y_test, y_pred_test)
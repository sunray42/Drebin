from sklearn.ensemble import AdaBoostClassifier
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

def adaboost(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{
            # 'base_estimator': [DecisionTreeClassifier(), MLPClassifier()],
            'n_estimators': [1000],
            'learning_rate': [0.3, 1, 3]
        }]

        scores = ['f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5, scoring=score, n_jobs=-1)
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
            pred_proba = clf.predict_proba(x_test)[:, 1]
    else:
        models = [
            AdaBoostClassifier(learning_rate=1, n_estimators=100)
        ]
        for model in models:
            print("Fitting AdaBoost ...")
            model.fit(x_train, y_train)
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            pred_proba = model.predict_proba(x_test)[:, 1]

    return y_pred_train, y_pred_test, pred_proba

if __name__ == "__main__":
    print("reading data ...")
    x_all, y_all = read.read()

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
    y_pred_train, y_pred_test, pred_proba_test = adaboost(x_train, y_train, x_test, y_test, TUNING=True)

    print("AdaBoost model's performance on training set:")
    evaluate(y_train, y_pred_train)
    print("")
    print("AdaBoost model's performance on testing set:")
    evaluate(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    fpr_adaboost, tpr_adaboost, threshold_adaboost = roc_curve(y_test, pred_proba_test)
    roc_auc = auc(fpr_adaboost, tpr_adaboost)
    plt.plot(fpr_adaboost, tpr_adaboost, color='darkorange', lw=2,
             label='adaboost ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curves')
    plt.legend(loc="lower right")
    plt.show()
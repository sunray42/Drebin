from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import read
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
# from main import evaluate

def evaluate(y_test, y_pred):
    print("Evaluating ...")
    print("Accuracy is %f." % accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Precision score is %f." % precision_score(y_test, y_pred))
    print("Recall score is %f." % recall_score(y_test, y_pred))
    print("F1 score is %f." % f1_score(y_test, y_pred))
    print("-----------------------------------")

def mnb(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{}]

        scores = ["accuracy", "f1"]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5, scoring=score, n_jobs=2)
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
            y_true, y_pred = y_test, clf.predict(x_test)
            print(classification_report(y_true, y_pred))
            print("")
            pred_proba = clf.predict_proba(x_test)[:, 1]

    else:
        models = [MultinomialNB()
                  ]
        for model in models:
            print("Fitting MNB ...")
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            pred_proba = model.predict_proba(x_test)[:, 1]
    return y_pred, pred_proba

if __name__ == "__main__":
    # 数据集划分
    print("Reading data ...")
    x_all, y_all = read.read()
    # x_all = np.delete(x_all, [4, 5], axis=1)        #delete feature 4 and feature 5, which have the lowest importances in rf
    # print(x_all[1])

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

    #MultinomialNaiveBayes
    y_pred_mnb, pred_proba_mnb = mnb(x_train, y_train, x_test, y_test)
    evaluate(y_test, y_pred_mnb)
    plt.figure(figsize=(10, 10))
    fpr_mnb, tpr_mnb, threshold_mnb = roc_curve(y_test, pred_proba_mnb)
    roc_auc = auc(fpr_mnb, tpr_mnb)
    plt.plot(fpr_mnb, tpr_mnb, color='darkorange', lw=2,
             label='GNB ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线


    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curves')
    plt.legend(loc="lower right")
    plt.show()
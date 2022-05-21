from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

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

def mlp(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{'alpha': [1e-2, 1e-1, 0.3],
                             'activation': ['logistic', 'tanh', 'relu'],    # 'identity',
                             'solver': ['lbfgs', 'adam']    # 'sgd',
                             }]

        scores = ["accuracy", "f1"]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(MLPClassifier(alpha=1e-4, activation='tanh'), tuned_parameters, cv=5, scoring=score,
                               n_jobs=2)
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
        models = [MLPClassifier(activation='tanh', alpha=0.00001, batch_size='auto', beta_1=0.9,
                                beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                hidden_layer_sizes=(100,), learning_rate='constant',
                                learning_rate_init=0.001, max_iter=200, momentum=0.9,
                                n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                                random_state=None, shuffle=True, solver='adam', tol=0.0001,
                                validation_fraction=0.1, verbose=False, warm_start=False)
                  ]
        for model in models:
            print("Fitting MLP ...")
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            pred_proba = model.predict_proba(x_test)[:, 1]

    return y_pred, pred_proba


if __name__ == '__main__':
    # 数据集划分
    print("Reading data ...")
    x_all, y_all = read.read()
    # x_all = np.delete(x_all, [4, 5], axis=1)        #delete feature 4 and feature 5, which have the lowest importances in rf
    # print(x_all[1])

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
    print('--------------------------------------------------------')
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_train[:10])
    print(x_test[:10])
    print('--------------------------------------------------------')

    # # 归一化
    # scaler = StandardScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)
    # print('--------------------------------------------------------')
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    # print(x_train[:10])
    # print(x_test[:10])
    # print('--------------------------------------------------------')

    y_pred_mlp, pred_proba_mlp = mlp(x_train, y_train, x_test, y_test, TUNING=True)
    evaluate(y_test, y_pred_mlp)
    # draw_roc(y_test, pred_proba)
    fpr_mlp, tpr_mlp, threshold_mlp = roc_curve(y_test, pred_proba_mlp)
    roc_auc = auc(fpr_mlp, tpr_mlp)
    plt.plot(fpr_mlp, tpr_mlp, color='green', lw=2,
             label='MLP ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curves')
    plt.legend(loc="lower right")
    plt.show()
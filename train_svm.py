from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def svm(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{'gamma': [1e-3, 1e-2, 1 / 8],
                             'C': [1, 10, 100, 1000]}]

        scores = ["accuracy", "f1"]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(SVC(kernel="rbf"), tuned_parameters, cv=5, scoring=score, n_jobs=2)
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
        models = [SVC(probability=True)
                  ]
        for model in models:
            print("Fitting SVM ...")
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            pred_proba = model.predict_proba(x_test)[:, 1]
    return y_pred, pred_proba

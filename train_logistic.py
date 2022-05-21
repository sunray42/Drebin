from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def logistic(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{'penalty': ['l2'], 'solver': ["lbfgs", "sag"], 'C': [0.01, 0.1, 1, 10, 100]},
                            {'penalty': ['l1'], 'solver': ["liblinear", "saga"], 'C': [0.01, 0.1, 1, 10, 100]}]

        scores = ["accuracy", "f1"]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(linear_model.LogisticRegression(), tuned_parameters, cv=5, scoring=score)
            clf.fit(x_train, y_train)

            print("Best parameters set found on development set:")
            print("")
            print(clf.best_estimator_)
            print(clf.best_params_)
            print("")
            print("Grid scores on development set:")
            print("")
            print(clf.best_score_)
            # for params, mean_score, scores in clf.cv_results_:
            #     print("%0.3f (+/-%0.03f) for %r"
            #           % (mean_score, scores.std() / 2, params))
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
        models = [
            linear_model.LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False)
        ]
        for model in models:
            print("Fitting logistic regression ...")
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            pred_proba = model.predict_proba(x_test)[:, 1]

    return y_pred, pred_proba
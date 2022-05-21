from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def rf(x_train, y_train, x_test, y_test, TUNING = False):
    if TUNING:

        tuned_parameters = [{'n_estimators': [10, 100, 1000],
                             'max_features': ["auto", "sqrt", "log2", None]}]

        scores = ["accuracy", "f1"]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score, n_jobs=2)
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
        models = [RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                         max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=1, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
                                         oob_score=False, random_state=None, verbose=0,
                                         warm_start=False),
                  ]
        for model in models:
            print("Fitting RF ...")
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            pred_proba = model.predict_proba(x_test)[:, 1]
    return y_pred, pred_proba
import read
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import train_logistic, train_mlp, train_rf, train_svm, train_dtc, train_mnb
import numpy as np

from train_knc import knc

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def draw_roc(y_test, pred_proba):
    #print(pred_proba)
    fpr, tpr, threshold = roc_curve(y_test, pred_proba)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()

def evaluate(y_test, y_pred):
    print("Evaluating ...")
    print("Accuracy is %f." % accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Precision score is %f." % precision_score(y_test, y_pred))
    print("Recall score is %f." % recall_score(y_test, y_pred))
    print("F1 score is %f." % f1_score(y_test, y_pred))
    print("-----------------------------------")

def classification_scatter(x_test, y_pred, model):      # 散点图
    print(x_test)
    print(y_pred)
    index = 0
    indices = np.argsort(y_pred)[::-1]
    y_pred = y_pred[indices]
    x_test = x_test[indices]
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            index = i + 1
    print(index)
    print(y_pred[0:index], y_pred[index:])
    x_1 = x_test[0:index, 1::6]
    x_0 = x_test[index:, 1::6]

    fig = plt.figure()
    plt.scatter(x_1[:, 0], x_1[:, 1], alpha=0.5)
    plt.scatter(x_0[:, 0], x_0[:, 1], c='green', alpha=0.6)
    plt.title(model)
    plt.show()

if __name__ == "__main__":
    # 数据集划分
    print("Reading data ...")
    x_all, y_all = read.read()
    # x_all = np.delete(x_all, [4, 5], axis=1)        #delete feature 4 and feature 5, which have the lowest importances in rf
    # print(x_all[1])

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # print(x_all.shape)
    # print(x_all[0])
    # x_new = SelectKBest(chi2, k=3).fit_transform(x_all, y_all)
    # print(x_new.shape)
    # print(x_new[0])
    #
    # clf = Pipeline([('anova', SelectPercentile(chi2)),
    #                 ('scaler', StandardScaler()),
    #                 ('svc', SVC(gamma="auto"))])
    #
    # score_means = list()
    # score_stds = list()
    # percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    #
    # for percentile in percentiles:
    #     clf.set_params(anova__percentile=percentile)
    #     this_scores = cross_val_score(clf, x_all, y_all)
    #     score_means.append(this_scores.mean())
    #     score_stds.append(this_scores.std())
    #
    # plt.errorbar(percentiles, score_means, np.array(score_stds))
    # plt.title(
    #     'Performance of the SVM-Anova varying the percentile of features selected')
    # plt.xticks(np.linspace(0, 100, 11, endpoint=True))
    # plt.xlabel('Percentile')
    # plt.ylabel('Accuracy Score')
    # plt.axis('tight')
    # plt.show()

    # logistic regression
    y_pred_lr, pred_proba_lr = train_logistic.logistic(x_train, y_train, x_test, y_test)
    evaluate(y_test, y_pred_lr)
    # draw_roc(y_test, pred_proba)
    plt.figure(figsize=(10, 10))
    fpr_lr, tpr_lr, threshold_lr = roc_curve(y_test, pred_proba_lr)
    roc_auc = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
             label='LR ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    # 散点图
    # classification_scatter(x_test, y_pred_lr, "logistic regression")

    # MLP
    y_pred_mlp, pred_proba_mlp = train_mlp.mlp(x_train, y_train, x_test, y_test)
    evaluate(y_test, y_pred_mlp)
    # draw_roc(y_test, pred_proba)
    fpr_mlp, tpr_mlp, threshold_mlp = roc_curve(y_test, pred_proba_mlp)
    roc_auc = auc(fpr_mlp, tpr_mlp)
    plt.plot(fpr_mlp, tpr_mlp, color='green', lw=2,
             label='MLP ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    # # 散点图
    # # classification_scatter(x_test, y_pred_mlp, "mlp")
    #
    # # svm
    # y_pred_svm, pred_proba_svm = train_svm.svm(x_train, y_train, x_test, y_test)
    # evaluate(y_test, y_pred_svm)
    # # draw_roc(y_test, pred_proba)
    # fpr_svm, tpr_svm, threshold_svm = roc_curve(y_test, pred_proba_svm)
    # roc_auc = auc(fpr_svm, tpr_svm)
    # plt.plot(fpr_svm, tpr_svm, color='skyblue', lw=2,
    #          label='SVM ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    #
    # # 散点图
    # # classification_scatter(x_test, y_pred_svm, "svm")
    #
    # # random forest
    # y_pred_rf, pred_proba_rf = train_rf.rf(x_train, y_train, x_test, y_test)
    # evaluate(y_test, y_pred_rf)
    # # draw_roc(y_test, pred_proba)
    # fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, pred_proba_rf)
    # roc_auc = auc(fpr_rf, tpr_rf)
    # plt.plot(fpr_rf, tpr_rf, color='red', lw=2,
    #          label='RF ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # #
    # # fpr, tpr, threshold = roc_curve(y_test, y_test)
    # # roc_auc = auc(fpr, tpr)
    # # plt.plot(fpr, tpr, color='yellow', lw=2,
    # #          label='RF ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    #
    # # 散点图
    # # classification_scatter(x_test, y_pred_rf, "random forest")
    #
    # # GaussionNaiveBayes
    # y_pred_gnb, pred_proba_gnb = train_mnb.mnb(x_train, y_train, x_test, y_test)
    # evaluate(y_test, y_pred_gnb)
    # fpr_gnb, tpr_gnb, threshold_gnb = roc_curve(y_test, pred_proba_gnb)
    # roc_auc = auc(fpr_gnb, tpr_gnb)
    # plt.plot(fpr_gnb, tpr_gnb, color='yellow', lw=2,
    #          label='GNB ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    #
    # # DesionTreeClassifier
    # y_pred_dtc, pred_proba_dtc = train_dtc.dtc(x_train, y_train, x_test, y_test)
    # evaluate(y_test, y_pred_dtc)
    # fpr_dtc, tpr_dtc, threshold_dtc = roc_curve(y_test, pred_proba_dtc)
    # roc_auc = auc(fpr_dtc, tpr_dtc)
    # plt.plot(fpr_dtc, tpr_dtc, color='darkblue', lw=2,
    #          label='DTC ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    #
    # # KNeighborsClassifier
    # y_pred_knc, pred_proba_knc = knc(x_train, y_train, x_test, y_test)
    # evaluate(y_test, y_pred_knc)
    # fpr_knc, tpr_knc, threshold_knc = roc_curve(y_test, pred_proba_knc)
    # roc_auc = auc(fpr_knc, tpr_knc)
    # plt.plot(fpr_knc, tpr_knc, color='purple', lw=2,
    #          label='KNC ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curves')
    plt.legend(loc="lower right")
    plt.show()
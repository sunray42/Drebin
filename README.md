# Drebin
Drebin malware analysis using machine learning and deep learning methods. We provide vectorized data in ./balanced_dastaset and ./imbalanced_dataset etc. [Raw data here.](https://www.sec.tu-bs.de/~danarp/drebin/)

# Introduction
9 machine learning methods and multilayer perceptron(MLP) are used to detect a malware in Drebin dataset.

# Results
|         method         | Precision | Recall | F1 score |
| :--------------------: | --------- | ------ | -------- |
| Logistic   regression  | 0.86      | 0.78   | 0.82     |
|          SVM           | 0.94      | 0.89   | 0.91     |
|     Random forest      | 0.93      | 0.94   | 0.94     |
|     Decision tree      | 0.90      | 0.93   | 0.92     |
|     MultinomialNB      | 0.65      | 0.83   | 0.73     |
|          KNN           | 0.91      | 0.95   | 0.93     |
|        AdaBoost        | 0.87      | 0.85   | 0.86     |
|       Perceptron       | 0.83      | 0.75   | 0.79     |
|  Gradient   Boosting   | 0.93      | 0.95   | 0.94     |
|          MLP           | 0.886     | 0.894  | 0.890    |
| 2_layer_neural_network | 0.84      | 0.86   | 0.85     |
| 3_layer_neural_network | 0.89      | 0.87   | 0.88     |
| 4_layer_neural_network | 0.90      | 0.92   | 0.91     |

Note:
* 9 machine learning methods and MLP are implemented in **sklearn**.
* 2_layer_neural_network/3_layer_neural_network/3_layer_neural_network are implemented exclusively in **numpy**(code in ./deeplearning). This implementation refers to [DeepLearn.ai by Andrew Ng.](https://github.com/enggen/Deep-Learning-Coursera)
* To deal with classification with imbalanced classes, we try the tools provided by **imblearn**. In addition, focal loss is used with the same purpose.
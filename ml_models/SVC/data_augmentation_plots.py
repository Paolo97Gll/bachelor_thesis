# #!/usr/bin/env python3
# # coding: utf-8
#
# # SVC: C-Support Vector Classification
# # Create data for confusion matrix, learning curve
# # and ROC curve for augmented data.
#
# # SAVE DATA IN A TXT OR HDF5 FILE! SO I CAN RETRY PLOT (FOR EXAMPLE WITH DIFFERENT SCALE)!
#
#
# # Scientific computing
# import numpy as np
# import pandas as pd
# from scipy import interp
#
# # Plot
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# #sns.set_context('paper')
#
# # Machine Learning
# # Model
# from sklearn.svm import SVC
# # Ensemble model
# from sklearn.ensemble import BaggingClassifier
# # Splitter Classes
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold
# # Splitter Functions
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import ShuffleSplit
# # Model validation
# from sklearn.model_selection import learning_curve
# # Training metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
#
# # Other
# import requests
# import threading
#
#
# # Best hyper-parameters
# best_kernel = 'rbf'
# best_gamma = 0.0145
# best_C = 0.8
#
# first_cycle = True
# with pd.HDFStore('../../classification/ris/OUT-classified-merged.h5', mode='r') as in_data:
#     for group in ['GLITCH', 'NO_GLITCH']:
#         if first_cycle == True:
#             data = np.array(in_data[group].to_numpy())
#             if group == 'GLITCH':
#                 target = np.ones(len(data))
#             elif group == 'NO_GLITCH':
#                 target = np.zeros(len(data))
#             else:
#                 print("ERROR.")
#             first_cycle = False
#         else:
#             data = np.concatenate((data, in_data[group].to_numpy()))
#             if group == 'GLITCH':
#                 target = np.concatenate((target, np.ones(len(in_data[group].to_numpy()))))
#             elif group == 'NO_GLITCH':
#                 target = np.concatenate((target, np.zeros(len(in_data[group].to_numpy()))))
#             else:
#                 print("ERROR.")
#
#
# ####################
# # Confusion Matrix #
# ####################
#
#
# # Confusion matrix function
# def plot_confusion_matrix(cm, labels, title, filename, normalize=True, save=True):
#     # Create DataFrame
#     df_cm = pd.DataFrame(cm, columns=labels, index=labels)
#     df_cm.index.name = 'True label'
#     df_cm.columns.name = 'Predicted label'
#     # Normalize
#     if normalize:
#         df_cm = df_cm.div(df_cm.sum(axis=1), axis=0).round(decimals=2)
#     with pd.HDFStore(filename) as out:
#         out.put(title, df_cm)
#     # Plot
#     fig, ax = plt.subplots()
#     ax = sns.heatmap(df_cm, cmap='Blues', annot=True)
#     axlim = ax.get_ylim()
#     ax.set_ylim(axlim[0] + 0.5, axlim[1] - 0.5)
#     if normalize:
#         ax.set_title('Confusion matrix (no multi glitch) (augmented)' + ' (normalized)')
#     else:
#         ax.set_title('Confusion matrix (no multi glitch) (augmented)')
#
#     if save == True:
#         fig.savefig('ris/plots/nmg_a-confusion_matrix.pdf')
#
#
# # Initialize classifier
# clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
#
# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0, stratify=target)
#
# # Apply data augmentation on training data
# X_train_aug, y_train_aug = X_train, y_train
# X_train_aug = np.concatenate((X_train_aug, -X_train))
# y_train_aug = np.concatenate((y_train_aug, y_train))
# for j in range(1,100):
#     X_train_aug = np.concatenate((X_train_aug, np.roll(X_train, j, axis=1)))
#     X_train_aug = np.concatenate((X_train_aug, -np.roll(X_train, j, axis=1)))
#     y_train_aug = np.concatenate((y_train_aug, y_train))
#     y_train_aug = np.concatenate((y_train_aug, y_train))
# # Apply data augmentation on testing data
# X_test_aug, y_test_aug = X_test, y_test
# X_test_aug = np.concatenate((X_test_aug, -X_test))
# y_test_aug = np.concatenate((y_test_aug, y_test))
# for j in range(1,100):
#     X_test_aug = np.concatenate((X_test_aug, np.roll(X_test, j, axis=1)))
#     X_test_aug = np.concatenate((X_test_aug, -np.roll(X_test, j, axis=1)))
#     y_test_aug = np.concatenate((y_test_aug, y_test))
#     y_test_aug = np.concatenate((y_test_aug, y_test))
#
# # Train and predict
# clf.fit(X_train_aug, y_train_aug)
# y_pred = clf.predict(X_test)
#
# # Compute confusion matrix
# labels = [0., 1.]
# labels_text = ['no glitch', 'glitch']
#
# # y_pred
# cm = confusion_matrix(y_test, y_pred, labels=labels)
# plot_confusion_matrix(cm, labels_text, 'ConfusionMatrix', 'ris/nmg-augmented_data_plot.h5', normalize=True)
#
#
# ##################
# # Learning Curve #
# ##################
#
#
# def plot_learning_curve(estimator, X, y, title, filename, ylim=None, xscale=None, cv=None,
#                         n_jobs=-1, n_curve_steps=20, save=True):
#     train_sizes = np.linspace(.1, 1.0, n_curve_steps)
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     if xscale is not None:
#         plt.xscale(xscale)
#     plt.xlabel('Training dataset size')
#     plt.ylabel('Score')
#     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#
#     with pd.HDFStore(filename) as out:
#         df = pd.DataFrame({'train_sizes': train_sizes, 'train_scores': train_scores, 'test_scores': test_scores})
#         out.put(title, df)
#
# # Classifier and cross-validation method
# clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
# cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
#
# data_aug = data
# data_aug = np.concatenate((data_aug, -data))
# target_aug = np.concatenate((target_aug, target))
# for j in range(1,100):
#     data_aug = np.concatenate((data_aug, np.roll(data, j, axis=1)))
#     data_aug = np.concatenate((data_aug, -np.roll(data, j, axis=1)))
#     target_aug = np.concatenate((target_aug, target))
#     target_aug = np.concatenate((target_aug, target))
#
# # Plot learning curve
# plot_learning_curve(clf, data, target, 'LearningCurve', 'ris/nmg-augmented_data_plot.h5', cv=cv, n_curve_steps=100)
#
#
# #############
# # ROC Curve #
# #############
#
#
# first_cycle = True
# with pd.HDFStore('../../classification/ris/OUT-classified-merged.h5', mode='r') as in_data:
#     for group in ['GLITCH', 'NO_GLITCH']:
#         if first_cycle == True:
#             data = np.array(in_data[group].to_numpy())
#             if group == 'GLITCH':
#                 target = np.ones(len(data))
#             elif group == 'NO_GLITCH':
#                 target = np.zeros(len(data))
#             else:
#                 print("ERROR.")
#             first_cycle = False
#         else:
#             data = np.concatenate((data, in_data[group].to_numpy()))
#             if group == 'GLITCH':
#                 target = np.concatenate((target, np.ones(len(in_data[group].to_numpy()))))
#             elif group == 'NO_GLITCH':
#                 target = np.concatenate((target, np.zeros(len(in_data[group].to_numpy()))))
#             else:
#                 print("ERROR.")
#
# def plot_roc_curve(estimator, X, y, cv, title, filename, save=True):
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     i = 0
#     for train, test in cv.split(X, y):
#         X_train, X_test = X[train], X[test]
#         y_train, y_test = y[train], y[test]
#
#         X_train_aug, y_train_aug = X_train, y_train
#         X_train_aug = np.concatenate((X_train_aug, -X_train))
#         y_train_aug = np.concatenate((y_train_aug, y_train))
#         for j in range(1,100):
#             X_train_aug = np.concatenate((X_train_aug, np.roll(X_train, j, axis=1)))
#             X_train_aug = np.concatenate((X_train_aug, -np.roll(X_train, j, axis=1)))
#             y_train_aug = np.concatenate((y_train_aug, y_train))
#             y_train_aug = np.concatenate((y_train_aug, y_train))
#         # Apply data augmentation on testing data
#         X_test_aug, y_test_aug = X_test, y_test
#         X_test_aug = np.concatenate((X_test_aug, -X_test))
#         y_test_aug = np.concatenate((y_test_aug, y_test))
#         for j in range(1,100):
#             X_test_aug = np.concatenate((X_test_aug, np.roll(X_test, j, axis=1)))
#             X_test_aug = np.concatenate((X_test_aug, -np.roll(X_test, j, axis=1)))
#             y_test_aug = np.concatenate((y_test_aug, y_test))
#             y_test_aug = np.concatenate((y_test_aug, y_test))
#
#         clas = estimator.fit(X_train_aug, y_train_aug)
#         probas_ = clas.predict_proba(X_test)
#         probas_aug_ = clas.predict_proba(X_test_aug_)
#         # Compute ROC curve and area the curve
#         fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#
#         i += 1
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(mean_fpr, mean_tpr, color='b',
#              label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')
#
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc="lower right")
#
#     if save == True:
#         plt.savefig(filename)
#
# # Classifier and cross-validation method
# clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, probability=True)
# cv = StratifiedKFold(n_splits=5)
#
# # Plot ROC curve
# plot_roc_curve(clf, data, target, cv, 'ROC curve (no multi glitch)', 'ris/plots/nmg-roc_curve.pdf')

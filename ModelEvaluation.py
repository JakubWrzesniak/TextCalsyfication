from copy import copy

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import seaborn as sns


def evaluate_model(name, clf, x_test, y_test, labels, params):
    predicted = clf.predict(x_test)
    clf_score = metrics.accuracy_score(y_test, predicted)
    cr = classification_report(y_test, predicted, target_names=labels)
    print(cr)
    f = open(name + "_" + str(clf.best_params_) + ".txt", "a")
    f.write(cr)
    f.close()
    print("clf score:", clf_score)
    confusion_matrix(clf, x_test, y_test, labels, name)
    for param in params:
        GridSearch_table_plot(clf, param, name)


def confusion_matrix(clf, x_test, y_test, labels, name):
    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        x_test,
        y_test,
        display_labels=labels,
        cmap=plt.cm.Blues,
        normalize='true',
    )
    disp.ax_.set_title(name + str(clf.best_params_))
    print(disp.confusion_matrix)
    plt.savefig(name + "_" + str(clf.best_params_) + '.png')


def GridSearch_table_plot(grid_clf, param_name, name,
                          graph=True,
                          display_all_params=True):
    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
    df.to_csv(name + param_name + ".csv", index=False)

    # plot
    if graph:
        # plt.figure(figsize=(8, 8))
        # plt.errorbar(params, means, yerr=stds)
        #
        # plt.axhline(y=best_mean + best_stdev, color='red')
        # plt.axhline(y=best_mean - best_stdev, color='red')
        # plt.plot(best_param, best_mean, 'or')
        #
        # plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        # plt.xlabel(param_name)
        # plt.ylabel('Score')
        xxd = [col for col in df if col.endswith('test_score') and col.startswith("split")]
        cols = copy(xxd)
        cols.insert(0, "param_" + param_name)
        print("df", df)
        print("Selected columns", cols)
        print("selected columns for score", xxd)
        df2 = df.loc[:, df.columns.isin(cols)]
        print("df2", df2)
        df3 = pd.melt(df2, id_vars=["param_" + param_name], value_vars=xxd)
        print(df3)
        plt.show()
        sns.boxplot(x=df3["param_" + param_name], y=df3['value']).set_title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.savefig(name + param_name + '.png')

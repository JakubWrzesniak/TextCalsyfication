from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ModelEvaluation import evaluate_model


def no_text_cleaning(X_train, X_test, y_train, y_test, selected_categories):
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("selector", SelectKBest(chi2, k=13000)),
        ("classifier", SVC())
    ])

    search_space = [
        {
            "classifier": [SVC()],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model("No Text Cleaning", clf, X_test, y_test, selected_categories, ["classifier__alpha"])


def tfidf_parameters(X_train, X_test, y_train, y_test, selected_categories):
    name = "Tfidf_parameters"
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("classifier", SVC(kernel="linear"))
    ])
    search_space = [
        {
            "classifier": [MultinomialNB()],
            'tfidf__max_df': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ["tfidf__max_df"])

    search_space = [
        {
            "classifier": [MultinomialNB()],
            'tfidf__min_df': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.9],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ['tfidf__min_df'])

    search_space = [
        {
            "classifier": [MultinomialNB()],
            'tfidf__use_idf': [True, False],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ['tfidf__use_idf'])

    search_space = [
        {
            "classifier": [MultinomialNB()],
            'tfidf__max_features': [10000, 13000, 15000, 17000, 20000],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ['tfidf__max_features'])


def multinominal_nb_alpha_test(X_train, X_test, y_train, y_test, selected_categories):
    name = "Alpha_Test"
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("selector", SelectKBest(chi2, k=13000)),
        ("classifier", MultinomialNB())
    ])

    search_space = [
        {
            "classifier": [MultinomialNB()],
            'classifier__alpha': [0.5, 0.2, 0.1, 1e-2, 1e-3, 0]
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ['classifier__alpha'])


def selector_chi(X_train, X_test, y_train, y_test, selected_categories):
    name = "Test selector"
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("selector", SelectKBest(chi2, k=1000)),
        ("classifier", SVC())
    ])

    search_space = [
        {
            "classifier": [SVC(kernel="linear")],
            "selector__k": [10000, 13000, 15000, 17000, 20000],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ["selector__k"])


def SVC_paramteres(X_train, X_test, y_train, y_test, selected_categories):
    name = "SVM parameters"
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("selector", SelectKBest(chi2, k=10000)),
        ("classifier", SVC(kernel="linear"))
    ])

    search_space = [
        {
            "classifier__C": [ 0.01, 0.1, 0.5, 0.8, 1, 1.2],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ['classifier__C'])


def SVC_VS_NB(X_train, X_test, y_train, y_test, selected_categories):
    name = "NB_vs_SVM"
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("selector", SelectKBest(chi2, k=15000)),
        ("classifier", SVC(kernel="linear"))
    ])

    search_space = [
        {
            "classifier": [SVC(kernel="linear"), MultinomialNB(alpha=0.01)],
        }
    ]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        return_train_score=True,
        n_jobs=-1,
        verbose=5
    )
    clf.fit(X_train, y_train)
    evaluate_model(name, clf, X_test, y_test, selected_categories, ["classifier"])


def cv_parameter_1(X_train, X_test, y_train, y_test, selected_categories):
    name = "cv_1"
    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("selector", SelectKBest(chi2, k=13000)),
        ("classifier", MultinomialNB(alpha=0.1))
    ])
    pipe.fit(X_train, y_train)
    print(pipe.score)
    disp = ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test,
        y_test,
        display_labels=selected_categories,
        cmap=plt.cm.Blues,
        normalize='true',
    )
    disp.ax_.set_title(name)
    print(disp.confusion_matrix)
    plt.savefig(name + "_" + str("noCV") + '.png')

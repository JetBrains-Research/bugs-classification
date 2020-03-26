import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer


def fuzzy_pr_auc_score(y, y_pred, **kwargs):
    results = []
    labels = kwargs['labels']

    def contains_label(marks, label):
        return label in marks.split('|')

    for predictions, target in zip(y_pred, y):
        most_probable_index = np.argmax(predictions)
        confidence = predictions[most_probable_index]
        is_true = contains_label(target, labels[most_probable_index])
        results.append((confidence, is_true))

    def count(predicate):
        return sum(predicate(*pair) for pair in results)

    def precision(treshold):
        positive = count(lambda x, a: x >= treshold)
        return 1 if positive == 0 else count(lambda x, a: (x >= treshold and a is True)) / positive

    def recall(treshold):
        return count(lambda x, a: (x >= treshold and a is True)) / y.shape[0]

    points = []
    for treshold in np.arange(0.05, 1.0, 0.05):
        points.append((precision(treshold), recall(treshold)))
    points = sorted(points, key=lambda point: point[1])

    auc = 0.0
    prev_precision = 1
    prev_recall = 0
    for point in points:
        auc += (point[1] - prev_recall) * (point[0] + prev_precision) / 2.0
        prev_precision = point[0]
        prev_recall = point[1]
    auc += (1 - prev_recall) * prev_precision / 2
    return auc


def get_cv_scores(clf, X, y, labels, n_splits=10):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    return cross_val_score(clf, X, y, cv=kfold,
                           scoring=make_scorer(fuzzy_pr_auc_score, needs_proba=True, labels=labels))


def get_pr_auc_score(clf, X_train, y_train, X_test, y_test, labels):
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    return fuzzy_pr_auc_score(y_test, y_pred, labels=labels)


def print_results(clf, X_train, y_train, X_test, y_test, labels):
    cv_scores = get_cv_scores(clf, X_train, y_train, labels)
    holdout_score = get_pr_auc_score(clf, X_train, y_train, X_test, y_test, labels)
    print(f'Cross validation scores:\n{cv_scores}\n')
    print(f'mean: {cv_scores.mean()}, std: {cv_scores.std()}')
    print(f'Holdout score:\n{holdout_score}')

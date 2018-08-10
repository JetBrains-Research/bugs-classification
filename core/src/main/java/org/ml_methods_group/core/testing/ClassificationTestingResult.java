package org.ml_methods_group.core.testing;

import java.util.Map;

import static org.ml_methods_group.core.CommonUtils.check;
import static org.ml_methods_group.core.CommonUtils.checkEquals;
import static org.ml_methods_group.core.CommonUtils.compose;

public class ClassificationTestingResult<V, M> implements TestingResults {
    private final Map<V, M> tests;
    private final Map<V, M> predictions;

    ClassificationTestingResult(Map<V, M> tests, Map<V, M> predictions) {
        this.tests = tests;
        this.predictions = predictions;
    }

    public double getRecall(M mark) {
        final long total = tests.entrySet()
                .stream()
                .map(Map.Entry::getValue)
                .filter(mark::equals)
                .count();
        final long truePositive = tests.entrySet()
                .stream()
                .filter(check(Map.Entry::getValue, mark::equals))
                .filter(check(compose(Map.Entry::getKey, predictions::get), mark::equals))
                .count();
        return (double) truePositive / total;
    }


    public double getPrecision(M mark) {
        final long predicted = predictions.entrySet()
                .stream()
                .map(Map.Entry::getValue)
                .filter(mark::equals)
                .count();
        final long truePositive = tests.entrySet()
                .stream()
                .filter(check(Map.Entry::getValue, mark::equals))
                .filter(check(compose(Map.Entry::getKey, predictions::get), mark::equals))
                .count();
        return (double) truePositive / predicted;
    }

    public double getAccuracy() {
        return (double) tests.entrySet()
                .stream()
                .filter(checkEquals(Map.Entry::getValue, compose(Map.Entry::getKey, predictions::get)))
                .count() / tests.size();
    }

    @Override
    public double getValue() {
        return getAccuracy();
    }
}

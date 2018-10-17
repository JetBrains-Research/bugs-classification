package org.ml_methods_group.testing;


import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.testing.validation.Validator;

import java.util.List;
import java.util.Map.Entry;

public class BasicClassificationTester<V, M> implements ClassificationTester<V, M> {
    private final List<V> tests;
    private final Validator<V, M> validator;

    public BasicClassificationTester(List<V> tests, Validator<V, M> validator) {
        this.tests = tests;
        this.validator = validator;
    }

    @Override
    public ClassificationTestingResult test(Classifier<V, M> classifier) {
        final ClassificationTestingResult results = new ClassificationTestingResult();
        for (V value : tests) {
            final Entry<M, Double> prediction = classifier.mostProbable(value);
            results.addTestResult(prediction.getValue(), validator.isValid(value, prediction.getKey()));
        }
        return results;
    }
}

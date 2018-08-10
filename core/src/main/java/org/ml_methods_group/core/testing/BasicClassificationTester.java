package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.Classifier;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class BasicClassificationTester<V, M> implements ClassificationTester<V, M> {
    private final Map<V, M> tests;

    public BasicClassificationTester(Map<V, M> tests) {
        this.tests = new HashMap<>(tests);
    }

    @Override
    public ClassificationTestingResult<V, M> test(Classifier<V, M> classifier) {
        final Map<V, M> predictions = tests.keySet()
                .stream()
                .collect(Collectors.toMap(Function.identity(), classifier::classify));
        return new ClassificationTestingResult<>(tests, predictions);
    }
}

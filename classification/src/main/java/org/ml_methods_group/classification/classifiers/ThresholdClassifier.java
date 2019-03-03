package org.ml_methods_group.classification.classifiers;

import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.MarkedClusters;

import java.util.Map;
import java.util.Optional;

public class ThresholdClassifier<T, M> implements Classifier<T, M> {

    private final Classifier<T, M> oracle;
    private final double threshold;

    public ThresholdClassifier(Classifier<T, M> oracle, double threshold) {
        this.oracle = oracle;
        this.threshold = threshold;
    }

    @Override
    public void train(MarkedClusters<T, M> samples) {
        oracle.train(samples);
    }

    @Override
    public Optional<M> classify(T value) {
        final Map.Entry<M, Double> bestMatch = mostProbable(value);
        return bestMatch.getValue() > threshold ? Optional.of(bestMatch.getKey()) : Optional.empty();
    }

    @Override
    public Map<M, Double> reliability(T value) {
        return oracle.reliability(value);
    }

    @Override
    public Map.Entry<M, Double> mostProbable(T value) {
        return oracle.mostProbable(value);
    }
}

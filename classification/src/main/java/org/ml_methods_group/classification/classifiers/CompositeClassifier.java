package org.ml_methods_group.classification.classifiers;

import org.ml_methods_group.common.*;

import java.util.Map;

public class CompositeClassifier<T, F, M> implements Classifier<T, M> {

    private final FeaturesExtractor<T, F> extractor;
    private final Classifier<Wrapper<F, T>, M> classifier;

    public CompositeClassifier(FeaturesExtractor<T, F> extractor, Classifier<Wrapper<F, T>, M> classifier) {
        this.extractor = extractor;
        this.classifier = classifier;
    }

    @Override
    public void train(MarkedClusters<T, M> samples) {
        classifier.train(samples.map(this::createWrapper));
    }

    @Override
    public M classify(T value) {
        return classifier.classify(createWrapper(value));
    }

    @Override
    public Map<M, Double> reliability(T value) {
        return classifier.reliability(createWrapper(value));
    }

    @Override
    public Map.Entry<M, Double> mostProbable(T value) {
        return classifier.mostProbable(createWrapper(value));
    }

    private Wrapper<F, T> createWrapper(T value) {
        return new Wrapper<>(extractor.process(value), value);
    }
}

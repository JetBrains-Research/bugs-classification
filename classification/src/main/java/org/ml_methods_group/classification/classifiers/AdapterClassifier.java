package org.ml_methods_group.classification.classifiers;

import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.MarkedClusters;

import java.util.Map;
import java.util.Optional;

public class AdapterClassifier<T, F, M> implements Classifier<T, M> {

    private final Classifier<F, M> classifier;
    private final FeaturesExtractor<T, F> extractor;

    public AdapterClassifier(Classifier<F, M> classifier, FeaturesExtractor<T, F> extractor) {
        this.classifier = classifier;
        this.extractor = extractor;
    }

    @Override
    public void train(MarkedClusters<T, M> samples) {
        classifier.train(samples.map(extractor::process));
    }

    @Override
    public Optional<M> classify(T value) {
        return classifier.classify(extractor.process(value));
    }

    @Override
    public Map<M, Double> reliability(T value) {
        return classifier.reliability(extractor.process(value));
    }

    @Override
    public Map.Entry<M, Double> mostProbable(T value) {
        return classifier.mostProbable(extractor.process(value));
    }
}

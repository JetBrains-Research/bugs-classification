package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.CommonUtils;
import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.Wrapper;

import java.util.Map;

public class CompositeClassifier<T, F, M> implements Classifier<T, M> {

    private final FeaturesExtractor<T, F> extractor;
    private final Classifier<Wrapper<F, T>, M> classifier;

    public CompositeClassifier(FeaturesExtractor<T, F> extractor, Classifier<Wrapper<F, T>, M> classifier) {
        this.extractor = extractor;
        this.classifier = classifier;
    }

    @Override
    public void train(Map<T, M> samples) {
        classifier.train(CommonUtils.mapKey(samples, Wrapper.wrap(extractor::process)));
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

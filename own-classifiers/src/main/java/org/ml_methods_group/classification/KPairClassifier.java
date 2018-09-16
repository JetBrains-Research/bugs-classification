package org.ml_methods_group.classification;

import org.ml_methods_group.core.*;

import java.util.List;
import java.util.Map;
import java.util.function.DoubleBinaryOperator;

public class KPairClassifier<V, F, M> implements Classifier<V, M> {

    private final List<V> options;
    private final int k;
    private final DistanceFunction<V> metric;
    private final DoubleBinaryOperator combiner;
    private final FeaturesExtractor<V, F> trainExtractor;
    private final BiFeaturesExtractor<V, V, F> testExtractor;
    private final Classifier<F, M> classifier;

    public KPairClassifier(List<V> options,
                           int k,
                           DistanceFunction<V> metric,
                           DoubleBinaryOperator combiner, FeaturesExtractor<V, F> trainExtractor,
                           BiFeaturesExtractor<V, V, F> testExtractor,
                           Classifier<F, M> classifier) {
        this.options = options;
        this.k = k;
        this.metric = metric;
        this.combiner = combiner;
        this.trainExtractor = trainExtractor;
        this.testExtractor = testExtractor;
        this.classifier = classifier;
    }

    @Override
    public void train(MarkedClusters<V, M> samples) {
        classifier.train(samples.map(trainExtractor::process));
    }

    @Override
    public Map<M, Double> reliability(V value) {
        final List<V> options = Utils.kNearest(value, this.options, k, metric);
        Map<M, Double> result = classifier.reliability(testExtractor.process(value, options.get(0)));
        for (V option : options.subList(1, options.size())) {
            final Map<M, Double> current = classifier.reliability(testExtractor.process(value, option));
            result = CommonUtils.combine(result, 0.0, current, 0.0, combiner::applyAsDouble);
        }
        return result;
    }
}

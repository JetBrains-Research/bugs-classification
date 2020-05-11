package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;

public class Approach<F> {
    public final FeaturesExtractor<Solution, F> extractor;
    public final DistanceFunction<F> metric;
    public final String name;

    public Approach(FeaturesExtractor<Solution, F> extractor, DistanceFunction<F> metric, String name) {
        this.extractor = extractor;
        this.metric = metric;
        this.name = name;
    }
}

package org.ml_methods_group.evaluation.approaches.clustering;

import org.ml_methods_group.clustering.clusterers.CompositeClusterer;
import org.ml_methods_group.clustering.clusterers.HAC;
import org.ml_methods_group.common.*;
import org.ml_methods_group.evaluation.approaches.Approach;

import java.util.function.Function;

public class ClusteringApproach {

    private final Function<Double, Clusterer<Solution>> creator;
    private final String name;

    public <T> ClusteringApproach(String name, Approach<T> approach) {
        this.name = name;
        this.creator = threshold -> new CompositeClusterer<>(approach.extractor, new HAC<>(
                threshold,
                1,
                CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
    }

    public Clusterer<Solution> getClusterer(double threshold) {
        return creator.apply(threshold);
    }

    public String getName() {
        return name;
    }
}

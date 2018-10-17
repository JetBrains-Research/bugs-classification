package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Clusterer;
import org.ml_methods_group.common.Solution;

import java.io.Serializable;
import java.util.List;

public class SolutionsClusterer implements Clusterer<Solution>, Serializable {

    private final Clusterer<Solution> clusterer;

    public SolutionsClusterer(Clusterer<Solution> clusterer) {
        this.clusterer = clusterer;
    }

    @Override
    public SolutionsClusters buildClusters(List<Solution> values) {
        return new SolutionsClusters(clusterer.buildClusters(values));
    }
}

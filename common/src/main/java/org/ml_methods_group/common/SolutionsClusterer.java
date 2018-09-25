package org.ml_methods_group.common;

import org.ml_methods_group.core.entities.Solution;

import java.util.List;

public class SolutionsClusterer implements Clusterer<Solution> {

    private final Clusterer<Solution> clusterer;

    public SolutionsClusterer(Clusterer<Solution> clusterer) {
        this.clusterer = clusterer;
    }

    @Override
    public SolutionsClusters buildClusters(List<Solution> values) {
        return new SolutionsClusters(clusterer.buildClusters(values));
    }
}

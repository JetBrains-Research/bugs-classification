package org.ml_methods_group.clustering.clusterers;

import org.ml_methods_group.common.DistanceFunction;

public class ClusterSizeLimitedHAC<T> extends HAC<T> {

    private final int maxClusterSize;

    public ClusterSizeLimitedHAC(double distanceLimit, int maxClusterSize, DistanceFunction<T> metric) {
        super(distanceLimit,0, metric);
        this.maxClusterSize = maxClusterSize;
    }

    protected void insertTripleIfNecessary(double distance, Community first, Community second) {
        if (first.entities.size() + second.entities.size() > maxClusterSize) {
            return;
        }
        super.insertTripleIfNecessary(distance, first, second);
    }
}

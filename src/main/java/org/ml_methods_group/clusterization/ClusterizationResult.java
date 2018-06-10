package org.ml_methods_group.clusterization;

import java.io.Serializable;
import java.util.List;

public class ClusterizationResult<T> implements Serializable {
    public final List<List<T>> clusters;
    public final DistanceFunction<T> distanceFunction;

    public ClusterizationResult(List<List<T>> clusters, DistanceFunction<T> distanceFunction) {
        this.clusters = clusters;
        this.distanceFunction = distanceFunction;
    }
}

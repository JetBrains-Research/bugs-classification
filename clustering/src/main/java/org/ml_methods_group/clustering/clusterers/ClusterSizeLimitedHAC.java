package org.ml_methods_group.clustering.clusterers;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.DistanceFunction;

import java.util.List;
import java.util.stream.Collectors;

public class ClusterSizeLimitedHAC<T> extends HAC<T> {

    private final int maxClusterSize;

    public ClusterSizeLimitedHAC(double distanceLimit, int maxClusterSize, DistanceFunction<T> metric) {
        super(distanceLimit, metric);
        this.maxClusterSize = maxClusterSize;
    }

    @Override
    public Clusters<T> buildClusters(List<T> values) {
        super.init(values);
        while (!heap.isEmpty()) {
            final Triple minTriple = heap.first();
            invalidateTriple(minTriple);
            final Community first = minTriple.first;
            final Community second = minTriple.second;
            mergeCommunities(first, second);
        }
        final List<Cluster<T>> clusters = communities.stream()
                .map(c -> c.entities)
                .map(Cluster::new)
                .collect(Collectors.toList());
        return new Clusters<>(clusters);
    }

    protected void insertTripleIfNecessary(double distance, Community first, Community second) {
        if (first.entities.size() > maxClusterSize  || second.entities.size() > maxClusterSize) {
            return;
        }
        super.insertTripleIfNecessary(distance, first, second);
    }
}

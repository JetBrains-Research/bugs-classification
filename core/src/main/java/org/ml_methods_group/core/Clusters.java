package org.ml_methods_group.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Clusters<V> implements Serializable {
    private final List<Cluster<V>> clusters;

    public Clusters(List<Cluster<V>> clusters) {
        this(new ArrayList<>(clusters));
    }

    private Clusters(ArrayList<Cluster<V>> clusters) {
        this.clusters = clusters;
    }

    public List<Cluster<V>> getClusters() {
        return Collections.unmodifiableList(clusters);
    }

    public <T> Clusters<T> map(Function<V, T> mapping) {
        final ArrayList<Cluster<T>> buffer = clusters.stream()
                .map(cluster -> cluster.map(mapping))
                .collect(Collectors.toCollection(ArrayList::new));
        return new Clusters<T>(buffer);
    }
}

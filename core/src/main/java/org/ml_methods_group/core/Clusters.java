package org.ml_methods_group.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Clusters<V> implements Serializable {
    private final List<Cluster<V>> clusters;

    public Clusters(List<Cluster<V>> clusters) {
        this.clusters = new ArrayList<>(clusters);
    }

    public List<Cluster<V>> getClusters() {
        return Collections.unmodifiableList(clusters);
    }
}

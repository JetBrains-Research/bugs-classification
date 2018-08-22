package org.ml_methods_group.core;

import java.util.List;

public interface Clusterer<V> {
    List<Cluster<V>> buildClusters(List<V> values);
}

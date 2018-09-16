package org.ml_methods_group.core;

import java.util.List;

public interface Clusterer<V> {
    Clusters<V> buildClusters(List<V> values);
}

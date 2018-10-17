package org.ml_methods_group.common;

import java.util.List;

public interface Clusterer<V> {
    Clusters<V> buildClusters(List<V> values);
}

package org.ml_methods_group.core;

import java.util.List;

public interface Clusterer<T> {
    List<List<T>> buildClusters(List<T> values);
}

package org.ml_methods_group.common;

import java.util.List;

public interface RepresentativesProducer<V, O> {
    List<O> getRepresentatives(Cluster<V> values);
    int getSelectionSize();
}

package org.ml_methods_group.common;

import java.util.List;

public interface RepresentativesPicker<V, O> {
    List<O> getRepresentatives(Cluster<V> values);
}

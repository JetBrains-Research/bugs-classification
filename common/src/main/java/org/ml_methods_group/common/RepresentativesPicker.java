package org.ml_methods_group.common;

import java.util.List;

public interface RepresentativesPicker<V> {
    List<V> getRepresentatives(Cluster<V> values);
}

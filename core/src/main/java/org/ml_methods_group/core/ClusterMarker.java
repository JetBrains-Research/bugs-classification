package org.ml_methods_group.core;

import java.util.List;

public interface ClusterMarker<V, M> {
    M mark(List<V> cluster);

    static <V, M> ClusterMarker<V, M> constMarker(M mark) {
        return x -> mark;
    }
}

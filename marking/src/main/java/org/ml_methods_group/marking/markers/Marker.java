package org.ml_methods_group.marking.markers;

import java.io.Serializable;

public interface Marker<V, M> {
    M mark(V value);

    static <V, M> Marker<V, M> constMarker(M mark) {
        return (Marker<V, M> & Serializable) x -> mark;
    }

    default Marker<V, M> or(Marker<V, M> other) {
        return (Marker<V, M> & Serializable) x -> {
            final M mark = mark(x);
            return mark != null ? mark : other.mark(x);
        };
    }
}

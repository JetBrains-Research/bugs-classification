package org.ml_methods_group.markers;

import java.util.function.Predicate;

public class PredicateMarker<V, M> implements Marker<V, M> {

    private final Marker<V, M> oracle;
    private final Predicate<V> predicate;
    private final M mark;

    public PredicateMarker(Predicate<V> predicate, M mark, Marker<V, M> oracle) {
        this.oracle = oracle;
        this.predicate = predicate;
        this.mark = mark;
    }

    @Override
    public M mark(V value) {
        return predicate.test(value) ? mark : oracle.mark(value);
    }
}

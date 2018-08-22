package org.ml_methods_group.core;

import java.util.List;
import java.util.function.Function;

public interface Unifier<V> {
    List<V> unify(List<V> values);

    static <V> Unifier<V> identity() {
        return x -> x;
    }
}

package org.ml_methods_group.common.preparation;

import java.util.List;

public interface Unifier<V> {
    List<V> unify(List<V> values);

    static <V> Unifier<V> identity() {
        return x -> x;
    }
}

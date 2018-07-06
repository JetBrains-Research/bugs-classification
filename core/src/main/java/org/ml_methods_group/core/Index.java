package org.ml_methods_group.core;

import java.util.Map;
import java.util.stream.Collectors;

public interface Index<T, V extends Comparable<? super V>> {
    void insert(T data, V value);

    Map<T, V> getIndex();

    default Map<T, V> getIndex(V lower, V upper) {
        return getIndex().entrySet()
                .stream()
                .filter(entry -> entry.getValue().compareTo(lower) >= 0)
                .filter(entry -> entry.getValue().compareTo(upper) < 0)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    void clean();
}

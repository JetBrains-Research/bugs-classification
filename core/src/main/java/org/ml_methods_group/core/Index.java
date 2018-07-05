package org.ml_methods_group.core;

import java.util.Map;
import java.util.stream.Collectors;

public interface Index<T> {
    void insert(T value, int count);

    Map<T, Integer> getIndex();

    default Map<T, Integer> getIndex(int lower, int upper) {
        return getIndex().entrySet()
                .stream()
                .filter(entry -> entry.getValue() >= lower)
                .filter(entry -> entry.getValue() < upper)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    void clean();
}

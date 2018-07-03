package org.ml_methods_group.core;

import java.util.Map;
import java.util.function.Function;

public interface IndexStorage {
    <K> void saveIndex(String name, Map<K, Long> index,
                          Function<K, String> keyToString);

    default <K> void saveIndex(String name, Map<K, Long> index) {
        saveIndex(name, index, Object::toString);
    }

    <K> Map<K, Long> loadIndex(String name,
                               Function<String, K> keyParser);

    default Map<String, Long> loadIndex(String name) {
        return loadIndex(name, Function.identity());
    }

    void dropIndex(String name);
}

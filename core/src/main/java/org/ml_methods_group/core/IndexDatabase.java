package org.ml_methods_group.core;

import java.util.Map;
import java.util.function.Function;

public interface IndexDatabase {
    <K, V> void saveIndex(String name, Map<K, V> index,
                          Function<K, String> keyToString,
                          Function<V, String> valueToString);

    default <K, V> void saveIndex(String name, Map<K, V> index) {
        saveIndex(name, index, Object::toString, Object::toString);
    }

    <K, V> Map<K, V> loadIndex(String name,
                               Function<String, K> keyParser,
                               Function<String, V> valueParser);

    default Map<String, String> loadIndex(String name) {
        return loadIndex(name, Function.identity(), Function.identity());
    }

    void dropIndex(String name);
}

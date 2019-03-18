package org.ml_methods_group.common;

import java.util.Optional;

public interface Repository<K, V> extends AutoCloseable {
    Optional<V> loadValue(K key);
    void storeValue(K key, V value);
}

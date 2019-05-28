package org.ml_methods_group.common;

public interface Database extends AutoCloseable {

    <K, V> Repository<K, V> repositoryForName(String name, Class<K> keyClass, Class<V> valueClass) throws Exception;
}

package org.ml_methods_group.testing.database;

public interface Database extends AutoCloseable {
    <T> Repository<T> getRepository(String name, Class<T> template);
    <T> Repository<T> getRepository(Class<T> template);
}

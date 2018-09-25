package org.ml_methods_group.testing.database;

import java.util.Iterator;
import java.util.Optional;

public interface Repository<T> extends Iterable<T> {
    void insert(T value);

    Iterator<T> get(Condition... conditions);

    Optional<T> find(Condition... conditions);

    Iterator<Proxy<T>> getProxy(Condition... conditions);

    void clear();

    ConditionSupplier conditionSupplier();
}

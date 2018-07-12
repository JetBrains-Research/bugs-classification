package org.ml_methods_group.core.database;

import java.util.Iterator;
import java.util.List;

public interface Repository<T, C extends Condition> extends Iterable<T> {
    void insert(T value);

    Iterator<T> get(List<C> conditions);

    Iterator<Proxy<T>> getProxy(List<C> conditions);

    void clear();

    ConditionSupplier<C> conditionSupplier();
}

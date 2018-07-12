package org.ml_methods_group.core.database;

public interface ConditionSupplier<T extends Condition> {
    T less(String field, long value);
    T less(String field, String value);
    T greater(String field, long value);
    T greater(String field, String value);
    T is(String field, long value);
    T is(String field, String value);
}

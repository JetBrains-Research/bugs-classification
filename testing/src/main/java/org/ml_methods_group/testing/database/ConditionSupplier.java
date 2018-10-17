package org.ml_methods_group.testing.database;

public interface ConditionSupplier {
    Condition less(String field, long value);
    Condition less(String field, Object value);
    Condition greater(String field, long value);
    Condition greater(String field, Object value);
    Condition is(String field, long value);
    Condition is(String field, Object value);
}

package org.ml_methods_group.core.database;

public interface ConditionSupplier {
    Condition less(String field, long value);
    Condition less(String field, String value);
    Condition greater(String field, long value);
    Condition greater(String field, String value);
    Condition is(String field, long value);
    Condition is(String field, String value);
}

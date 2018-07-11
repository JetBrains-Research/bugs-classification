package org.ml_methods_group.database;

import org.ml_methods_group.core.database.ConditionSupplier;

public class SQLConditionSupplier implements ConditionSupplier<SQLCondition> {

    private static final SQLConditionSupplier INSTANCE = new SQLConditionSupplier();

    private SQLConditionSupplier() {
    }

    public SQLCondition less(String column, long value) {
        return new SQLCondition(column + " < " + value);
    }

    public SQLCondition less(String column, String value) {
        return new SQLCondition(column + " < '" + value + "'");
    }

    public SQLCondition greater(String column, long value) {
        return new SQLCondition(column + " > " + value);
    }

    public SQLCondition greater(String column, String value) {
        return new SQLCondition(column + " > '" + value + "'");
    }

    public SQLCondition is(String column, long value) {
        return new SQLCondition(column + " = " + value);
    }

    public SQLCondition is(String column, String value) {
        return new SQLCondition(column + " = '" + value + "'");
    }

    public static SQLConditionSupplier instance() {
        return INSTANCE;
    }
}

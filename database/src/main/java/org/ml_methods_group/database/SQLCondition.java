package org.ml_methods_group.database;

import org.ml_methods_group.core.database.Condition;

class SQLCondition implements Condition {
    private final String condition;

    SQLCondition(String condition) {
        this.condition = condition;
    }

    String toSQL() {
        return condition;
    }
}

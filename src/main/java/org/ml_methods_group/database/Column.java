package org.ml_methods_group.database;

public class Column {
    public final String name;
    public final Type type;
    public final boolean isKey;

    public Column(String name, Type type) {
        this.name = name;
        this.type = type;
        isKey = false;
    }

    public Column(String name, Type type, boolean isKey) {
        this.name = name;
        this.type = type;
        this.isKey = isKey;
    }

    public String toSQL() {
        return name + " " + type + " NOT NULL" + (isKey ? " PRIMARY KEY" : "");
    }
}

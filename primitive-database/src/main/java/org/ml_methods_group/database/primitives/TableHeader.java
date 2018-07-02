package org.ml_methods_group.database.primitives;

public class TableHeader {
    final Column[] columns;
    final String table;

    public TableHeader(String table, Column... columns) {
        this.columns = columns;
        this.table = table;
    }
}

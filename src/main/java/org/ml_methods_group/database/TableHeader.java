package org.ml_methods_group.database;

public class TableHeader {
    final Column[] columns;
    final String table;

    TableHeader(String table, Column... columns) {
        this.columns = columns;
        this.table = table;
    }
}

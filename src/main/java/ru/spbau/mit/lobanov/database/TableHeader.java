package ru.spbau.mit.lobanov.database;

public class TableHeader {
    final Column[] columns;
    final String table;

    TableHeader(String table, Column... columns) {
        this.columns = columns;
        this.table = table;
    }
}

package org.ml_methods_group.database;

import static org.ml_methods_group.database.DataType.SERIAL;

class Column {
    static final Column ID = new Column("ID", SERIAL, true);

    private final String name;
    private final DataType type;
    private final boolean isKey;

    private Column(String name, DataType type, boolean isKey) {
        this.name = name;
        this.type = type;
        this.isKey = isKey;
    }

    Column(String name, DataType type) {
        this(name, type, false);
    }

    String getName() {
        return name;
    }

    DataType getType() {
        return type;
    }

    boolean isKey() {
        return isKey;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Column column = (Column) o;

        return isKey == column.isKey && name.equals(column.name) && type == column.type;
    }

    @Override
    public int hashCode() {
        int result = name.hashCode();
        result = 31 * result + type.hashCode();
        result = 31 * result + (isKey ? 1 : 0);
        return result;
    }
}

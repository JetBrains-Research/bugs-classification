package org.ml_methods_group.database;

import org.ml_methods_group.core.IndexDatabase;
import org.ml_methods_group.database.primitives.*;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

public class BasicIndexDatabase implements IndexDatabase {

    private final Database database;

    public BasicIndexDatabase(Database database) {
        this.database = database;
        try {
            database.createTable(Tables.codes_header);
            database.createTable(Tables.diff_header);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public <K, V> void saveIndex(String name, Map<K, V> index, Function<K, String> keyToString, Function<V, String> valueToString) {
        try {
            final TableHeader header = crateHeader(name);
            database.dropTable(header);
            database.createTable(header);
            final Table table = database.getTable(header);
            for (Entry<K, V> entry : index.entrySet()) {
                table.insert(new Object[]{keyToString.apply(entry.getKey()), valueToString.apply(entry.getValue())});
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public <K, V> Map<K, V> loadIndex(String name, Function<String, K> keyParser, Function<String, V> valueParser) {
        try {
            final TableHeader header = crateHeader(name);
            final Table table = database.getTable(header);
            final Map<K, V> index = new HashMap<>();
            for (Table.ResultWrapper wrapper : table) {
                index.put(
                        keyParser.apply(wrapper.getStringValue("key")),
                        valueParser.apply(wrapper.getStringValue("value"))
                );
            }
            return index;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void dropIndex(String name) {
        try {
            final TableHeader header = crateHeader(name);
            database.dropTable(header);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public static TableHeader crateHeader(String name) {
        return new TableHeader("index_database_" + name,
                new Column("key", Type.TEXT, true),
                new Column("value", Type.TEXT));
    }
}

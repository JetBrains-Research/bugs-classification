package org.ml_methods_group.database;

import org.ml_methods_group.core.IndexStorage;
import org.ml_methods_group.database.primitives.*;
import sun.security.pkcs11.wrapper.Functions;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

public class BasicIndexStorage implements IndexStorage {

    private final Database database;

    public BasicIndexStorage() {
        this(new Database());
    }

    public BasicIndexStorage(Database database) {
        this.database = database;
    }

    @Override
    public <K> void saveIndex(String name, Map<K, Long> index, Function<K, String> keyToString) {
        try {
            final TableHeader header = crateHeader(name);
            database.dropTable(header);
            database.createTable(header);
            final Table table = database.getTable(header);
            for (Entry<K, Long> entry : index.entrySet()) {
                table.insert(new Object[]{keyToString.apply(entry.getKey()), entry.getValue().intValue()});
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public <K> Map<K, Long> loadIndex(String name, Function<String, K> keyParser) {
        try {
            final TableHeader header = crateHeader(name);
            final Table table = database.getTable(header);
            final Map<K, Long> index = new HashMap<>();
            for (Table.ResultWrapper wrapper : table) {
                index.put(
                        keyParser.apply(wrapper.getStringValue("key")),
                        (long) wrapper.getIntValue("value")
                );
            }
            return index;
        } catch (Exception e) {
            return Collections.emptyMap();
        }
    }

    @Override
    public void dropIndex(String name) {
        final TableHeader header = crateHeader(name);
        database.dropTable(header);

    }

    public static TableHeader crateHeader(String name) {
        return new TableHeader("index_database_" + name,
                new Column("key", Type.TEXT, true),
                new Column("value", Type.INTEGER));
    }
}

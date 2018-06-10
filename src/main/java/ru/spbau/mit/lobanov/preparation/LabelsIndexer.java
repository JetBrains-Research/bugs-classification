package ru.spbau.mit.lobanov.preparation;

import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;

import java.io.IOException;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

public class LabelsIndexer {
    public static void indexLabels(Database database) throws SQLException, IOException {
        database.dropTable(Tables.labels_header);
        database.createTable(Tables.labels_header);
        final Table diffs = database.getTable(Tables.diff_header);
        final Table labels = database.getTable(Tables.labels_header);
        final Map<String, Integer> index = new HashMap<>();
        for (Table.ResultWrapper diff : diffs) {
            final String label = diff.getStringValue("label");
            final String oldLabel = diff.getStringValue("old_label");
            PreparationUtils.incrementCounter(index, label);
            PreparationUtils.incrementCounter(index, oldLabel);
        }
        int idGenerator = 1;
        for (Map.Entry<String, Integer> entry : index.entrySet()) {
            labels.insert(new Object[]{entry.getKey(), idGenerator++, entry.getValue()});
        }
    }

    // word -> id
    public static Map<String, Integer> getLabels(Database database) throws SQLException {
        final Table labels = database.getTable(Tables.labels_header);
        final Map<String, Integer> index = new HashMap<>();
        for (Table.ResultWrapper item : labels) {
            final String label = item.getStringValue("label");
            final int id = item.getIntValue("id");
            index.put(label, id);
        }
        return index;
    }

    public static void main(String[] args) throws IOException, SQLException {
        try (Database database = new Database()) {
            indexLabels(database);
        }
    }
}

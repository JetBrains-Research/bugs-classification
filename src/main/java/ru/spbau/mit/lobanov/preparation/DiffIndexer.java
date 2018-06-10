package ru.spbau.mit.lobanov.preparation;

import ru.spbau.mit.lobanov.FileUtils;
import ru.spbau.mit.lobanov.changes.AtomicChange;
import ru.spbau.mit.lobanov.changes.ChangeUtils;
import ru.spbau.mit.lobanov.changes.EncodingStrategy;
import ru.spbau.mit.lobanov.changes.EncodingStrategy.ChangeAttribute;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static ru.spbau.mit.lobanov.changes.EncodingStrategy.ChangeAttribute.*;
import static ru.spbau.mit.lobanov.preparation.PreparationUtils.incrementCounter;


public class DiffIndexer {

    public static void indexDiffs(Database database, EncodingStrategy... strategies) throws SQLException, IOException {
        database.dropTable(Tables.diff_types_header);
        database.createTable(Tables.diff_types_header);
        final Table diffsTypes = database.getTable(Tables.diff_types_header);
        final Table diffs = database.getTable(Tables.diff_header);
        final HashMap<EncodingStrategy, Map<Long, Integer>> counters = new HashMap<>();
        for (Table.ResultWrapper diff : diffs) {
            final AtomicChange change = ChangeUtils.fromData(diff.asArray(), 1);
            for (EncodingStrategy strategy : strategies) {
                incrementCounter(counters.computeIfAbsent(strategy, x -> new HashMap<>()), strategy.encode(change));
            }
        }
        final Object[] buffer = new Object[diffsTypes.columnCount()];
        final Map<Long, Integer> total = new HashMap<>();
        for (Map.Entry<EncodingStrategy, Map<Long, Integer>> entry : counters.entrySet()) {
            final EncodingStrategy strategy = entry.getKey();
            final Map<Long, Integer> index = entry.getValue();
            for (Map.Entry<Long, Integer> result : index.entrySet()) {
                final long type = result.getKey();
                final int count = result.getValue();
                final Integer oldValue = total.put(type, count);
                if (oldValue == null) {
                    total.put(type, count);
                    storeData(buffer, type, count, strategy);
                    diffsTypes.insert(buffer);
                } else if (oldValue != count) {
                    throw new RuntimeException("Conflict results");
                }
            }
        }
    }

    public static Map<Long, Integer> getIndex(int problem, Database database, EncodingStrategy... strategies) throws SQLException {
        final Table codes = database.getTable(Tables.codes_header);
        final Table diffs = database.getTable(Tables.diff_header);
        final HashMap<EncodingStrategy, Map<Long, Integer>> counters = new HashMap<>();
        final Set<Integer> sessionIds = new HashSet<>();
        final Iterator<Table.ResultWrapper> codesIterator = codes.find("problem", problem);
        while (codesIterator.hasNext()){
            final Table.ResultWrapper session = codesIterator.next();
            final String submitId = session.getStringValue("id");
            final int id = Integer.parseInt(submitId.substring(0, submitId.length() - 2));
            sessionIds.add(id);
        }
        for (Table.ResultWrapper diff : diffs) {
            if (!sessionIds.contains(diff.getIntValue("session_id"))) {
                continue;
            }
            final AtomicChange change = ChangeUtils.fromData(diff.asArray(), 1);
            for (EncodingStrategy strategy : strategies) {
                incrementCounter(counters.computeIfAbsent(strategy, x -> new HashMap<>()), strategy.encode(change));
            }
        }
        final Map<Long, Integer> total = new HashMap<>();
        for (Map.Entry<EncodingStrategy, Map<Long, Integer>> entry : counters.entrySet()) {
            final Map<Long, Integer> index = entry.getValue();
            for (Map.Entry<Long, Integer> result : index.entrySet()) {
                final long type = result.getKey();
                final int count = result.getValue();
                final Integer oldValue = total.put(type, count);
                if (oldValue == null) {
                    total.put(type, count);
                } else if (oldValue != count) {
                    throw new RuntimeException("Conflict results");
                }
            }
        }
        return total;
    }

    public static Map<Long, Integer> getIndex(int problem, Database database) throws SQLException, FileNotFoundException {
        return getIndex(problem, database, getDefaultStrategies());
    }

    // diff code -> count
    public static Map<Long, Integer> getIndex(Database database) throws SQLException {
        final Table labels = database.getTable(Tables.diff_types_header);
        final Map<Long, Integer> index = new HashMap<>();
        for (Table.ResultWrapper item : labels) {
            final long code = item.getBigIntValue("id");
            final int count = item.getIntValue("count");
            index.put(code, count);
        }
        return index;
    }

    private static void storeData(Object[] buffer, long code, int count, EncodingStrategy strategy) {
        buffer[0] = code;
        for (ChangeAttribute attribute : ChangeAttribute.values()) {
            buffer[1 + attribute.ordinal()] = strategy.decodeAttribute(code, attribute);
        }
        buffer[buffer.length - 1] = count;
    }

    public static EncodingStrategy[] getDefaultStrategies() throws FileNotFoundException {
        final Map<String, Integer> dictionary = FileUtils.readDictionary("dictionary.txt",
                Function.identity(), Integer::parseInt);
        return new EncodingStrategy[]{
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, OLD_PARENT_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_PARENT_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, LABEL_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_PARENT_TYPE, LABEL_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_PARENT_TYPE, LABEL_TYPE,
                        OLD_LABEL_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE,
                        OLD_PARENT_TYPE, LABEL_TYPE, OLD_LABEL_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE,
                        OLD_PARENT_TYPE, OLD_PARENT_OF_PARENT_TYPE),
                new EncodingStrategy(dictionary, CHANGE_TYPE, NODE_TYPE, LABEL_TYPE, OLD_LABEL_TYPE)
        };
    }

    public static void main(String[] args) throws SQLException, IOException {
        try (Database database = new Database()) {
            indexDiffs(database, getDefaultStrategies());
        }
    }
}

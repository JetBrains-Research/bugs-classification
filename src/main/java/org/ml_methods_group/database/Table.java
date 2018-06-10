package org.ml_methods_group.database;

import com.google.common.collect.ImmutableSet;

import java.io.UnsupportedEncodingException;
import java.sql.*;
import java.util.Collections;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Table implements Iterable<Table.ResultWrapper> {
    private final Connection connection;
    private final String table;
    private final String insertTemplate;
    private final Column[] columns;
    private final int idIndex;

    public Table(Connection connection, TableHeader header) {
        this.connection = connection;
        this.table = header.table;
        this.columns = header.columns;
        idIndex = IntStream.range(0, header.columns.length)
                .filter(i -> header.columns[i].isKey)
                .findFirst()
                .orElse(-1);

        final String typesDeclaration = Stream.of(columns)
                .map(column -> column.name).collect(Collectors.joining(", "));
        final String templates = Collections.nCopies(columns.length, "?")
                .stream()
                .collect(Collectors.joining(", "));
        insertTemplate = "INSERT INTO " + table +
                " (" + typesDeclaration +
                ") VALUES (" + templates + ")";
    }

    public void insert(Object[] values) throws SQLException, UnsupportedEncodingException {
        try (PreparedStatement statement = connection.prepareStatement(insertTemplate)) {
            for (int i = 0; i < values.length; i++) {
                switch (columns[i].type) {
                    case TEXT:
                        statement.setString(i + 1, (String) values[i]);
                        break;
                    case INTEGER:
                        statement.setInt(i + 1, (Integer) values[i]);
                        break;
                    case BIGINT:
                        statement.setLong(i + 1, (Long) values[i]);
                        break;
                    case FLOAT:
                        statement.setDouble(i + 1, (Double) values[i]);
                        break;
                    case BYTEA:
                        statement.setBytes(i + 1, ((String) values[i]).getBytes("UTF-8"));
                        break;
                }
            }
            statement.execute();
        }
    }

    public void delete(Object id) throws SQLException {
        final String request = "DELETE FROM " + table + " WHERE " +
                table + "." + columns[idIndex].name + " = '" + id + "';";
        try (Statement dataQuery = connection.createStatement()) {
            dataQuery.execute(request);
        }
    }

    public ResultWrapper findFirst(Object id) throws SQLException, UnsupportedEncodingException {
        final ResultWrapper result = findFirstOrNull(id);
        if (result == null) {
            throw new NoSuchElementException();
        }
        return result;
    }

    public ResultWrapper findFirstOrNull(Object id) throws SQLException, UnsupportedEncodingException {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + columns[idIndex].name + " = '" + id + "';";
        try (PreparedStatement query = connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            return rs.next() ? new ResultWrapper(rs) : null;
        }
    }

    public Iterator<ResultWrapper> find(int columnId, Object value) throws SQLException {
        return find(columns[columnId].name, value);
    }

    public Iterator<ResultWrapper> find(String column, Object value) throws SQLException {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + column + " = '" + value + "';";
        final PreparedStatement query = connection.prepareStatement(request);
        final ResultSet resultSet = query.executeQuery();
        return new Iterator<ResultWrapper>() {
            private ResultWrapper next;

            {
                tryReadNext();
            }

            @Override
            public boolean hasNext() {
                return next != null;
            }

            @Override
            public ResultWrapper next() {
                final ResultWrapper result = next;
                tryReadNext();
                return result;
            }

            private void tryReadNext() {
                try {
                    if (resultSet.next()) {
                        next = new ResultWrapper(resultSet);
                    } else {
                        next = null;
                        resultSet.close();
                        query.close();
                    }
                } catch (Exception e) {
                    try {
                        resultSet.close();
                        query.close();
                    } catch (SQLException ignored) {
                    }
                    next = null;
                }
            }
        };
    }

    public boolean containsKey(Object id) throws SQLException {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + columns[idIndex].name + " = '" + id + "';";
        try (PreparedStatement query =
                     connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            return rs.next();
        }
    }

    public int size() throws SQLException {
        final String request = "SELECT COUNT(1) FROM " + table + ";";
        try (PreparedStatement query =
                     connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            rs.next();
            return rs.getInt(1);
        }
    }

    public int columnCount() {
        return columns.length;
    }

    public Iterator<ResultWrapper> listAll() throws SQLException {
        final String request = "SELECT * FROM " + table + ";";
        final PreparedStatement query = connection.prepareStatement(request);
        final ResultSet resultSet = query.executeQuery();
        return new Iterator<ResultWrapper>() {
            private ResultWrapper next;

            {
                tryReadNext();
            }

            @Override
            public boolean hasNext() {
                return next != null;
            }

            @Override
            public ResultWrapper next() {
                final ResultWrapper result = next;
                tryReadNext();
                return result;
            }

            private void tryReadNext() {
                try {
                    if (resultSet.next()) {
                        next = new ResultWrapper(resultSet);
                    } else {
                        next = null;
                        resultSet.close();
                        query.close();
                    }
                } catch (Exception e) {
                    try {
                        resultSet.close();
                        query.close();
                    } catch (SQLException ignored) {
                    }
                    next = null;
                }
            }
        };
    }

    private int getIndex(String columnName) {
        return IntStream.range(0, columns.length)
                .filter(i -> columns[i].name.equals(columnName))
                .findFirst()
                .orElse(-1);
    }

    @Override
    public Iterator<ResultWrapper> iterator() {
        try {
            return listAll();
        } catch (Exception e) {
            return ImmutableSet.<ResultWrapper>of().iterator();
        }
    }

    public class ResultWrapper {
        private final Object[] results;

        public ResultWrapper(Object[] results) {
            this.results = results;
        }

        public ResultWrapper(ResultSet resultSet) throws SQLException, UnsupportedEncodingException {
            results = new Object[columns.length];
            for (int i = 0; i < columns.length; i++) {
                switch (columns[i].type) {
                    case TEXT:
                        results[i] = resultSet.getString(i + 1);
                        break;
                    case INTEGER:
                        results[i] = resultSet.getInt(i + 1);
                        break;
                    case BIGINT:
                        results[i] = resultSet.getLong(i + 1);
                        break;
                    case FLOAT:
                        results[i] = resultSet.getDouble(i + 1);
                        break;
                    case BYTEA:
                        results[i] = new String(resultSet.getBytes(i + 1), "UTF-8");
                        break;
                }
            }
        }

        public Object[] asArray() {
            return results;
        }

        public double getDoubleValue(String columnName) throws SQLException {
            final int index = getIndex(columnName);
            return getDoubleValue(index);
        }

        public double getDoubleValue(int index) throws SQLException {
            switch (columns[index].type) {
                case FLOAT:
                    return (Double) results[index];
                case INTEGER:
                    return (Integer) results[index];
                case BIGINT:
                    return (Long) results[index];
                default:
                    return Double.valueOf((String) results[index]);
            }
        }

        public String getStringValue(String columnName) throws SQLException {
            final int index = getIndex(columnName);
            return getStringValue(index);
        }

        public String getStringValue(int index) throws SQLException {
            return results[index].toString();
        }

        public int getIntValue(String columnName) throws SQLException {
            final int index = getIndex(columnName);
            return getIntValue(index);
        }

        public int getIntValue(int index) throws SQLException {
            if (columns[index].type == Type.FLOAT) {
                throw new RuntimeException("Cant cast float to int");
            } else if (columns[index].type == Type.INTEGER) {
                return (Integer) results[index];
            } else {
                return Integer.valueOf((String) results[index]);
            }
        }

        public long getBigIntValue(String columnName) throws SQLException {
            final int index = getIndex(columnName);
            return getBigIntValue(index);
        }

        public long getBigIntValue(int index) throws SQLException {
            if (columns[index].type == Type.FLOAT) {
                throw new RuntimeException("Cant cast float to int");
            } else if (columns[index].type == Type.INTEGER) {
                return (Integer) results[index];
            } else if (columns[index].type == Type.BIGINT) {
                return (Long) results[index];
            } else {
                return Integer.valueOf((String) results[index]);
            }
        }
    }
}

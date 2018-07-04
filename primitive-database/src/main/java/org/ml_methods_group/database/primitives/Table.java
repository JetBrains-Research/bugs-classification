package org.ml_methods_group.database.primitives;

import com.google.common.collect.ImmutableSet;
import org.ml_methods_group.database.DatabaseException;

import java.io.UnsupportedEncodingException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collections;
import java.util.Iterator;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

@SuppressWarnings("unused")
public class Table implements Iterable<Table.ResultWrapper> {
    private final Connection connection;
    private final String table;
    private final String insertTemplate;
    private final Column[] columns;
    private final int idIndex;

    Table(Connection connection, TableHeader header) {
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

    public void insert(Object[] values) {
        try (PreparedStatement statement = connection.prepareStatement(insertTemplate)) {
            for (int i = 0; i < values.length; i++) {
                switch (columns[i].type) {
                    case TEXT:
                        statement.setString(i + 1, values[i].toString());
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
                        statement.setBytes(i + 1, (values[i].toString()).getBytes("UTF-8"));
                        break;
                }
            }
            statement.execute();
        } catch (SQLException | UnsupportedEncodingException e) {
            throw new DatabaseException(e);
        }
    }

    public Optional<ResultWrapper> findFirst(String column, Object value) {
        return Optional.ofNullable(findFirstOrNull(column, value));
    }

    public ResultWrapper findFirstOrNull(String column, Object value) {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + column + " = '" + value + "';";
        return findFirstOrNull(request);
    }

    public Optional<ResultWrapper> findFirst(String column1, Object value1, String column2, Object value2) {
        return Optional.ofNullable(findFirstOrNull(column1, value1, column2, value2));
    }

    public ResultWrapper findFirstOrNull(String column1, Object value1, String column2, Object value2) {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + column1 + " = '" + value1 + "' AND " +
                table + "." + column2 + " = '" + value2 + "';";
        return findFirstOrNull(request);
    }

    private ResultWrapper findFirstOrNull(String request) {
        try (PreparedStatement query = connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            return rs.next() ? new ResultWrapper(rs) : null;
        } catch (SQLException | UnsupportedEncodingException e) {
            throw new DatabaseException(e);
        }
    }

    public Iterator<ResultWrapper> find(int columnId, Object value) {
        return find(columns[columnId].name, value);
    }

    public Iterator<ResultWrapper> find(String column, Object value) {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + column + " = '" + value + "';";
        return find(request);
    }

    public Iterator<ResultWrapper> find(String column1, Object value1, String column2, Object value2) {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + column1 + " = '" + value1 + "' AND " +
                table + "." + column2 + " = '" + value2 + "';";
        return find(request);
    }

    private Iterator<ResultWrapper> find(String request) {
        try {
            final PreparedStatement query;
            query = connection.prepareStatement(request);
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
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    public boolean containsKey(Object id) {
        final String request = "SELECT * FROM " + table + " WHERE " +
                table + "." + columns[idIndex].name + " = '" + id + "';";
        try (PreparedStatement query =
                     connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            return rs.next();
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    public int size() {
        final String request = "SELECT COUNT(1) FROM " + table + ";";
        try (PreparedStatement query =
                     connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            rs.next();
            return rs.getInt(1);
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    public int columnCount() {
        return columns.length;
    }

    public Iterator<ResultWrapper> listAll() {
        final String request = "SELECT * FROM " + table + ";";
        return find(request);
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

        public double getDoubleValue(String columnName) {
            final int index = getIndex(columnName);
            return getDoubleValue(index);
        }

        public double getDoubleValue(int index) {
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

        public <T extends Enum<T>> T getEnumValue(String columnName, Class<T> enumType) {
            final int index = getIndex(columnName);
            return getEnumValue(index, enumType);
        }

        public <T extends Enum<T>> T getEnumValue(int index, Class<T> enumType) {
            switch (columns[index].type) {
                case TEXT:
                    return Enum.valueOf(enumType, (String) results[index]);
                default:
                    throw new RuntimeException("String type expected");
            }
        }

        public String getStringValue(String columnName) {
            final int index = getIndex(columnName);
            return getStringValue(index);
        }

        public String getStringValue(int index) {
            return results[index].toString();
        }

        public int getIntValue(String columnName) {
            final int index = getIndex(columnName);
            return getIntValue(index);
        }

        public int getIntValue(int index) {
            if (columns[index].type == Type.FLOAT) {
                throw new DatabaseException("Cant cast float to int");
            } else if (columns[index].type == Type.INTEGER) {
                return (Integer) results[index];
            } else {
                return Integer.valueOf((String) results[index]);
            }
        }

        public long getBigIntValue(String columnName) {
            final int index = getIndex(columnName);
            return getBigIntValue(index);
        }

        public long getBigIntValue(int index) {
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

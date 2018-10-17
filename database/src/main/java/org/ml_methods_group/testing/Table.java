package org.ml_methods_group.testing;

import org.ml_methods_group.testing.database.Condition;

import java.io.UnsupportedEncodingException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.sql.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

class Table {
    private final Connection connection;
    private final String insertTemplate;
    private final String selectTemplate;
    private final String name;
    private final List<Column> columns;

    Table(Connection connection, String name, List<Column> columns) {
        this.connection = connection;
        this.name = name;
        this.columns = columns;
        final String typesDeclaration = columns.stream()
                .filter(column -> !column.getType().service)
                .map(Column::getName)
                .collect(Collectors.joining(", "));
        final String templates = columns.stream()
                .filter(column -> !column.getType().service)
                .map(x -> "?")
                .collect(Collectors.joining(", "));
        insertTemplate = "INSERT INTO " + name +
                " (" + typesDeclaration +
                ") VALUES (" + templates + ")";
        selectTemplate = "SELECT " + columns.stream()
                .map(Column::getName)
                .collect(Collectors.joining(", ")) +
                " FROM " + name;
    }

    void insert(List<Object> data) {
        if (data.size() != columns.size()) {
            throw new IllegalArgumentException();
        }
        try (PreparedStatement statement = connection.prepareStatement(insertTemplate)) {
            int pointer = 1;
            for (int i = 0; i < columns.size(); i++) {
                final Column column = columns.get(i);
                switch (column.getType()) {
                    case ENUM:
                        statement.setString(pointer++, data.get(i).toString());
                        break;
                    case STRING:
                        statement.setBytes(pointer++, data.get(i).toString().getBytes(StandardCharsets.UTF_16));
                        break;
                    case INTEGER:
                        statement.setInt(pointer++, (Integer) data.get(i));
                        break;
                    case LONG:
                        statement.setLong(pointer++, (Long) data.get(i));
                        break;
                    case DOUBLE:
                        statement.setDouble(pointer++, (Double) data.get(i));
                        break;
                    case BOOLEAN:
                        statement.setBoolean(pointer++, (Boolean) data.get(i));
                        break;
                }
            }
            statement.execute();
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    Iterator<ResultWrapper> select(Condition... conditions) {
        final String condition = Arrays.stream(conditions)
                .map(Object::toString)
                .collect(Collectors.joining(" AND ", " ", " "));
        final String request = selectTemplate + (conditions.length == 0 ? "" : " WHERE" + condition) + ";";
        try {
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
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    Optional<ResultWrapper> find(Condition... conditions) {
        final String condition = Arrays.stream(conditions)
                .map(Object::toString)
                .collect(Collectors.joining(" AND ", " ", " "));
        final String request = selectTemplate + (conditions.length == 0 ? "" : " WHERE " + condition) + " LIMIT 1;";
        try (PreparedStatement query = connection.prepareStatement(request);
             ResultSet rs = query.executeQuery()) {
            return rs.next() ? Optional.of(new ResultWrapper(rs)) : Optional.empty();
        } catch (SQLException | UnsupportedEncodingException e) {
            throw new DatabaseException(e);
        }
    }

    void create() {
        final String request = columns.stream()
                .map(Table::createColumnDeclaration)
                .collect(Collectors.joining(",\n",
                        "CREATE TABLE IF NOT EXISTS " + name + "(\n",
                        "\n);"));
        execute(request);
    }

    void clear() {
        final String request = "DELETE FROM " + name + ";";
        execute(request);
    }

    void drop() {
        final String request = "DROP TABLE IF EXISTS " + name + ";";
        execute(request);
    }

    private int columnIndex(Column column) {
        for (int i = 0; i < columns.size(); i++) {
            if (columns.get(i).equals(column)) {
                return i;
            }
        }
        return -1;
    }

    private void execute(String request) {
        try (Statement statement = connection.createStatement()) {
            statement.execute(request);
        } catch (Exception e) {
            throw new DatabaseException(e);
        }
    }

    private static String createColumnDeclaration(Column column) {
        return column.getName() + " " + column.getType().sqlName +
                (column.isKey() ? " PRIMARY KEY" : "") + " NOT NULL";
    }

    class ResultWrapper {
        private final Object[] results;

        private ResultWrapper(ResultSet resultSet) throws SQLException, UnsupportedEncodingException {
            results = new Object[columns.size()];
            for (int i = 0; i < columns.size(); i++) {
                results[i] = resultSet.getObject(i + 1);
            }
        }

        double getDoubleValue(Column column) {
            final int index = columnIndex(column);
            switch (columns.get(index).getType()) {
                case DOUBLE:
                    return (Double) results[index];
                case INTEGER:
                    return (Integer) results[index];
                case LONG:
                    return (Long) results[index];
                default:
                    return Double.valueOf((String) results[index]);
            }
        }

        <T> T getEnumValue(Column column, Class<T> enumType) {
            final int index = columnIndex(column);
            switch (columns.get(index).getType()) {
                case ENUM:
                case STRING:
                    final String value = (String) results[index];
                    return value.equals("null") ? null : parseEnum(value, enumType);
                default:
                    throw new RuntimeException("String type expected");
            }
        }

        String getStringValue(Column column) {
            final int index = columnIndex(column);
            switch (columns.get(index).getType()) {
                case STRING:
                    return new String((byte[]) results[index], StandardCharsets.UTF_16);
                default:
                    return results[index].toString();
            }
        }

        int getIntValue(Column column) {
            final int index = columnIndex(column);
            switch (columns.get(index).getType()) {
                case DOUBLE:
                    throw new DatabaseException("Cant cast float to int");
                case INTEGER:
                    return (Integer) results[index];
                case LONG:
                    return Math.toIntExact((Long) results[index]);
                default:
                    return Integer.valueOf((String) results[index]);
            }
        }

        long getLongValue(Column column) {
            final int index = columnIndex(column);
            switch (columns.get(index).getType()) {
                case DOUBLE:
                    throw new RuntimeException("Cant cast float to int");
                case INTEGER:
                    return (Integer) results[index];
                case LONG:
                    return (Long) results[index];
                default:
                    return Long.valueOf((String) results[index]);
            }
        }

        boolean getBooleanValue(Column column) {
            final int index = columnIndex(column);
            switch (columns.get(index).getType()) {
                case BOOLEAN:
                    return (Boolean) results[index];
                case INTEGER:
                    return (Integer) results[index] != 0;
                case LONG:
                    return (Long) results[index] != 0L;
                default:
                    throw new RuntimeException("Cant cast to boolean");
            }
        }

        Object getValue(Column column, Class<?> template) {
            if (template.equals(Double.class) || template.equals(double.class)) {
                return getDoubleValue(column);
            } else if (template.equals(Integer.class) || template.equals(int.class)) {
                return getIntValue(column);
            } else if (template.equals(Long.class) || template.equals(long.class)) {
                return getLongValue(column);
            } else if (template.equals(Boolean.class) || template.equals(boolean.class)) {
                return getBooleanValue(column);
            } else if (template.equals(String.class)) {
                return getStringValue(column);
            } else if (Enum.class.isAssignableFrom(template)) {
                return getEnumValue(column, template);
            }
            throw new DatabaseException("Unsupported data type requested: " + template.getCanonicalName());
        }

    }

    private static <T> T parseEnum(String token, Class<T> template) {
        if (!Enum.class.isAssignableFrom(template)) {
            throw new IllegalArgumentException();
        }
        try {
            final Method method = template.getMethod("valueOf", String.class);
            //noinspection unchecked
            return (T) method.invoke(null, token);
        } catch (Exception e) {
            throw new DatabaseException(e);
        }
    }
}

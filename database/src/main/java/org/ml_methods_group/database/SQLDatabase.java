package org.ml_methods_group.database;

import org.ml_methods_group.testing.database.Database;
import org.ml_methods_group.testing.database.Repository;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.List;

public class SQLDatabase implements Database {

    static {
        try {
            DriverManager.registerDriver(new org.postgresql.Driver());
        } catch (SQLException ignored) {
        }
    }

    private final Connection connection;

    public SQLDatabase() {
        this("jdbc:postgresql://localhost/database", "myRole", "");
    }

    public SQLDatabase(String uri, String user, String password) {
        try {
            connection = DriverManager.getConnection(uri, user, password);
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    Table getTable(String name, List<Column> columns) {
        return new Table(connection, name, columns);
    }

    @Override
    public void close() {
        try {
            connection.close();
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    @Override
    public <T> Repository<T> getRepository(String name, Class<T> template) {
        return new SQLRepository<>(name, template, this);
    }

    @Override
    public <T> Repository<T> getRepository(Class<T> template) {
        return new SQLRepository<>(template, this);
    }
}

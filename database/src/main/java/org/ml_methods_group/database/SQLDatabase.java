package org.ml_methods_group.database;

import org.ml_methods_group.core.database.Database;
import org.ml_methods_group.core.database.Repository;
import org.postgresql.Driver;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.List;

public class SQLDatabase implements Database {

    private final Connection connection;

    public SQLDatabase() {
        try {
            DriverManager.registerDriver(new Driver());
            connection = DriverManager
                    .getConnection("jdbc:postgresql://localhost/database?user=myRole&ssl=false");
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

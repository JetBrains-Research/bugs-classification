package org.ml_methods_group.database;

import org.postgresql.Driver;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.List;

public class Database implements AutoCloseable {

    private final Connection connection;

    public Database() {
        try {
            DriverManager.registerDriver(new Driver());
            connection = DriverManager
                    .getConnection("jdbc:postgresql://localhost/database?user=myRole&ssl=false");
        } catch (SQLException e) {
            throw new DatabaseException(e);
        }
    }

    public Table getTable(String name, List<Column> columns) {
        return new Table(connection, name, columns);
    }

    @Override
    public void close() throws Exception {
        connection.close();
    }
}

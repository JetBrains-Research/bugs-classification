package org.ml_methods_group.database.primitives;/*
 * Copyright 2017 Machine Learning Methods in Software Engineering Group of JetBrains Research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import org.postgresql.Driver;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Database implements AutoCloseable {

    private final Connection connection;

    public Database() throws SQLException {
        DriverManager.registerDriver(new Driver());
        connection = DriverManager
                .getConnection("jdbc:postgresql://localhost/database?user=myRole&ssl=false");
    }

    public void createTable(TableHeader header) throws SQLException {
        final StringBuilder request = new StringBuilder("CREATE TABLE IF NOT EXISTS " + header.table + " (\n");
        final String columnsDeclarations = Stream.of(header.columns)
                .map(Column::toSQL)
                .collect(Collectors.joining(",\n"));
        request.append(columnsDeclarations);
        request.append("\n);");
        try (Statement dataQuery = connection.createStatement()) {
            dataQuery.execute(request.toString());
        }
    }

    public void dropTable(TableHeader header) throws SQLException {
        try (Statement dataQuery = connection.createStatement()) {
            dataQuery.execute("DROP TABLE IF EXISTS " + header.table + ";");
        }
    }

    public Table getTable(TableHeader header) {
        return new Table(connection, header);
    }

    public void close() throws SQLException {
        connection.close();
    }
}

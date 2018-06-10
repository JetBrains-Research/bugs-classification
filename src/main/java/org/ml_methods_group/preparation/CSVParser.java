package org.ml_methods_group.preparation;

import au.com.bytecode.opencsv.CSVReader;
import org.ml_methods_group.database.Database;
import org.ml_methods_group.database.Tables;
import org.ml_methods_group.database.Table;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.sql.SQLException;

public class CSVParser {

    private static final char UNEXCITING_SYMBOL = (char) 255;

    private static void insertCodes(Database database) throws IOException, SQLException {
        try (InputStream resourceStream = CSVParser.class.getResourceAsStream("/submissions.csv");
             InputStreamReader resourceStreamReader = new InputStreamReader(resourceStream);
             CSVReader reader = new CSVReader(resourceStreamReader, ',', '\"', UNEXCITING_SYMBOL)) {
            database.dropTable(Tables.codes_header);
            database.createTable(Tables.codes_header);
            Table table = database.getTable(Tables.codes_header);
            reader.readNext(); // read header
            while (true) {
                String[] data = reader.readNext();
                if (data == null) {
                    break;
                }
                final String id = data[0] + "_" + data[3];
                table.insert(new Object[]{id, data[5], Integer.valueOf(data[1])});
            }
        }
    }

    private static void insertProblems(Database database) {
        try (InputStream resourceStream = CSVParser.class.getResourceAsStream("/submissions.csv");
             InputStreamReader resourceStreamReader = new InputStreamReader(resourceStream);
             CSVReader reader = new CSVReader(resourceStreamReader, ',', '\"', UNEXCITING_SYMBOL)) {
            database.dropTable(Tables.problems_header);
            database.createTable(Tables.problems_header);
            Table table = database.getTable(Tables.problems_header);
            reader.readNext();
            while (true) {
                String[] data = reader.readNext();
                if (data == null) {
                    break;
                }
                final Integer id = Integer.valueOf(data[1]);
                if (!table.containsKey(id))
                    table.insert(new Object[]{id, data[2]});
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws IOException, SQLException {
        try (Database database = new Database()) {
            insertCodes(database);
            insertProblems(database);
        }
    }
}

package ru.spbau.mit.lobanov.preparation;

import au.com.bytecode.opencsv.CSVReader;
import javafx.application.Application;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.sql.SQLException;

import static ru.spbau.mit.lobanov.database.Tables.codes_header;
import static ru.spbau.mit.lobanov.database.Tables.problems_header;

public class CSVParser {

    private static final char UNEXCITING_SYMBOL = (char) 255;

    private static void insertCodes(Database database) throws IOException, SQLException {
        try (InputStream resourceStream = CSVParser.class.getResourceAsStream("/submissions.csv");
             InputStreamReader resourceStreamReader = new InputStreamReader(resourceStream);
             CSVReader reader = new CSVReader(resourceStreamReader, ',', '\"', UNEXCITING_SYMBOL)) {
            database.dropTable(codes_header);
            database.createTable(codes_header);
            Table table = database.getTable(codes_header);
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
            database.dropTable(problems_header);
            database.createTable(problems_header);
            Table table = database.getTable(problems_header);
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

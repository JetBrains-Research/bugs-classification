package ru.spbau.mit.lobanov.view;

import difflib.DiffUtils;
import difflib.Patch;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;

import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class CodeView {

    public static void main(String[] args) throws SQLException, UnsupportedEncodingException {
        try (Database database = new Database();
             Scanner scanner = new Scanner(System.in)) {
            final Table codes = database.getTable(Tables.codes_header);
            while (scanner.hasNextInt()) {
                ViewUtils.printDiff(codes, scanner.nextInt());
            }
        }
    }
}

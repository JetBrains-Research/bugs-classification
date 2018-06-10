package org.ml_methods_group.view;

import org.ml_methods_group.database.Database;
import org.ml_methods_group.database.Table;
import org.ml_methods_group.database.Tables;

import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
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

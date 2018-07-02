package org.ml_methods_group.core.preparation;

import au.com.bytecode.opencsv.CSVReader;
import org.ml_methods_group.core.Solution.Verdict;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.basic.BasicSolution;

import java.io.*;

public class CSVParser {
    private static final char UNEXCITING_SYMBOL = (char) 255;

    public static void parse(File file, SolutionDatabase database) throws IOException {
        database.clear();
        database.create();
        try (InputStream resourceStream = new FileInputStream(file);
             InputStreamReader resourceStreamReader = new InputStreamReader(resourceStream);
             CSVReader reader = new CSVReader(resourceStreamReader, ',', '\"', UNEXCITING_SYMBOL)) {
            reader.readNext(); // read header
            while (true) {
                String[] data = reader.readNext();
                if (data == null) {
                    break;
                }
                database.insertSolution(new BasicSolution(data[5], Integer.parseInt(data[1]), Integer.parseInt(data[0]),
                        data[3].charAt(0) == '1' ? Verdict.OK : Verdict.FAIL));
            }
        }
    }
}

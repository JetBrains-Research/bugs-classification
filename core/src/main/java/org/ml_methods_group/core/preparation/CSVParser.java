package org.ml_methods_group.core.preparation;

import au.com.bytecode.opencsv.CSVReader;
import org.ml_methods_group.core.entities.Solution.Verdict;

import java.io.*;
import java.util.NoSuchElementException;

public class CSVParser {
    private static final char UNEXCITING_SYMBOL = (char) 255;
    private final int[] indexes = new int[Column.values().length];
    private final CSVReader reader;
    private String[] buffer;
    private String[] next;


    public CSVParser(File file) throws IOException {
        this(new FileInputStream(file));
    }

    public CSVParser(InputStream stream) throws IOException {
        reader = new CSVReader(new InputStreamReader(stream), ',', '\"', UNEXCITING_SYMBOL);
        parseHeader(reader.readNext());
        next = reader.readNext();
    }

    private void parseHeader(String[] header) {
        for (int i = 0; i < header.length; i++) {
            indexes[Column.byName(header[i].trim()).ordinal()] = i;
        }
    }

    private void readLine() throws IOException {
        buffer = next;
        if (next != null) {
            next = reader.readNext();
        }
    }

    private String getToken(Column column) {
        if (buffer == null) {
            throw new IllegalStateException("Parser isn't ready!");
        }
        return buffer[indexes[column.ordinal()]];
    }

    public void nextLine() throws IOException {
        if (next != null) {
            readLine();
        } else {
            throw new NoSuchElementException();
        }
    }

    public boolean hasNextLine() {
        return next != null;
    }

    public String getCode() {
        return getToken(Column.CODE);
    }

    public String getProblemText() {
        return getToken(Column.PROBLEM_TEXT);
    }

    public String getLanguage() {
        return getToken(Column.LANGUAGE);
    }

    public int getSessionId() {
        return Integer.parseInt(getToken(Column.SESSION_ID));
    }

    public int getProblemId() {
        return Integer.parseInt(getToken(Column.PROBLEM_ID));
    }

    public Verdict getVerdict() {
        return getToken(Column.VERDICT).charAt(0) == '0' ? Verdict.FAIL : Verdict.OK;
    }

    private enum Column {
        SESSION_ID("\\S*data_id\\S*"),
        PROBLEM_ID("\\S*step_id\\S*"),
        PROBLEM_TEXT("\\S*step_text\\S*"),
        VERDICT("\\S*status\\S*"),
        LANGUAGE("\\S*language\\S*"),
        CODE("\\S*code\\S*");

        final String name;

        Column(String name) {
            this.name = name;
        }

        static Column byName(String name) {
            for (Column column : values()) {
                if (name.matches(column.name)) {
                    return column;
                }
            }
            throw new NoSuchElementException(name);
        }
    }
}

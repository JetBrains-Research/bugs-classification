package org.ml_methods_group.parsing;

import au.com.bytecode.opencsv.CSVReader;
import jdk.nashorn.internal.parser.Token;

import java.io.*;
import java.util.EnumMap;
import java.util.NoSuchElementException;
import java.util.Optional;

public class CSVParser<T extends Enum<T>> {
    private static final char UNEXCITING_SYMBOL = (char) 255;
    private final EnumMap<T, Integer> indexes;
    private final CSVReader reader;
    private String[] buffer;
    private String[] next;


    public CSVParser(File file, ColumnMatcher<T> matcher, Class<T> columns) throws IOException {
        this(new FileInputStream(file), matcher, columns);
    }

    public CSVParser(InputStream stream, ColumnMatcher<T> matcher, Class<T> columns) throws IOException {
        reader = new CSVReader(new InputStreamReader(stream), ',', '\"', UNEXCITING_SYMBOL);
        indexes = new EnumMap<>(columns);
        parseHeader(reader.readNext(), indexes, matcher);
        next = reader.readNext();
    }

    private static <T extends Enum<T>> void parseHeader(String[] header, EnumMap<T, Integer> storage,
                                                        ColumnMatcher<T> matcher) {
        for (int i = 0; i < header.length; i++) {
            final Optional<T> column = matcher.match(header[i]);
            if (column.isPresent()) {
                storage.put(column.get(), i);
            }
        }
    }

    private void readLine() throws IOException {
        buffer = next;
        if (next != null) {
            next = reader.readNext();
        }
    }

    public String getToken(T column) {
        if (buffer == null) {
            throw new IllegalStateException("Parser isn't ready!");
        }
        final int index = indexes.getOrDefault(column, -1);
        if (index == -1) {
            throw new NoSuchElementException();
        }
        return buffer[index];
    }

    public String getTokenOrDefault(T column, String value) {
        if (buffer == null) {
            throw new IllegalStateException("Parser isn't ready!");
        }
        final int index = indexes.getOrDefault(column, -1);
        if (index == -1) {
            return value;
        }
        return buffer[index];
    }

    public int getInt(T column) {
        return Integer.parseInt(getToken(column));
    }

    public int getIntOrDefault(T column, int value) {
        final String token = getTokenOrDefault(column, null);
        return token == null ? value : Integer.parseInt(getToken(column));
    }

    public long getLong(T column) {
        return Long.parseLong(getToken(column));
    }

    public long getLongOrDefault(T column, long value) {
        final String token = getTokenOrDefault(column, null);
        return token == null ? value : Long.parseLong(getToken(column));
    }

    public boolean getBoolean(T column) {
        final String token = getToken(column);
        if (token.matches("-?\\d+")) {
            return Integer.parseInt(token) != 0;
        } else {
            return Boolean.parseBoolean(token);
        }
    }

    public <E extends Enum<E>> E getEnum(T column, Class<E> template) {
        return Enum.valueOf(template, getToken(column));
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

    public interface ColumnMatcher<T extends Enum<T>> {
        Optional<T> match(String columnName);
    }
}

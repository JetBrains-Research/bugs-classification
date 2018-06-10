package org.ml_methods_group;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import java.util.function.Function;


public class FileUtils {
    public static <T> List<T> readValues(String filename, Function<String, T> parser) throws FileNotFoundException {
        final List<T> result = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(filename))) {
            while (scanner.hasNext()) {
                result.add(parser.apply(scanner.next()));
            }
        }
        return result;
    }

    public static List<String> readTokens(String filename) throws FileNotFoundException {
        return readValues(filename, Function.identity());
    }

    public static <T> void writeValues(String filename, Iterable<T> values) throws FileNotFoundException {
        writeValues(filename, values, Object::toString);
    }

    public static <T> void writeValues(String filename, Iterable<T> values,
                                       Function<T, String> translator) throws FileNotFoundException {
        try(PrintWriter out = new PrintWriter(new File(filename))) {
            for (T value : values) {
                out.println(translator.apply(value));
            }
        }
    }

    public static Map<String, String> readDictionary(String filename) throws FileNotFoundException {
        return readDictionary(filename, Function.identity(), Function.identity());
    }

    public static <K, V> Map<K, V> readDictionary(String filename, Function<String, K> keyParser,
                                                  Function<String, V> valueParser) throws FileNotFoundException {
        final Map<K, V> result = new HashMap<>();
        try (Scanner scanner = new Scanner(new File(filename))) {
            while (scanner.hasNext()) {
                final K key = keyParser.apply(nextNotEmptyLine(scanner));
                final V value = valueParser.apply(nextNotEmptyLine(scanner));
                result.put(key, value);
            }
        }
        return result;
    }

    public static <K, V> void writeDictionary(String filename, Map<K, V> dictionary) throws FileNotFoundException {
        writeDictionary(filename, dictionary, Object::toString, Object::toString);
    }

    public static <K, V> void writeDictionary(String filename, Map<K, V> dictionary, Function<K, String> keyTranslator,
                                              Function<V, String> valueTranslator) throws FileNotFoundException {
        try(PrintWriter out = new PrintWriter(new File(filename))) {
            for (Map.Entry<K, V> entry : dictionary.entrySet()) {
                out.println(keyTranslator.apply(entry.getKey()));
                out.println(valueTranslator.apply(entry.getValue()));
                out.println();
            }
        }
    }

    private static String nextNotEmptyLine(Scanner scanner) {
        String line = scanner.nextLine().trim();
        while (scanner.hasNextLine() && line.isEmpty()) {
            line = scanner.nextLine().trim();
        }
        return line.isEmpty() ? null : line;
    }
}

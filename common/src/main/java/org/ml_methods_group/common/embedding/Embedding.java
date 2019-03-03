package org.ml_methods_group.common.embedding;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.function.Function;

public class Embedding<T, K> {
    private final Map<K, double[]> embeddings;
    private final Function<T, K> keyExtractor;
    private final double[] defaultValue;

    public Embedding(Map<K, double[]> embeddings, Function<T, K> keyExtractor,
                     K defaultKey) throws FileNotFoundException {
        this.embeddings = embeddings;
        this.keyExtractor = keyExtractor;
        this.defaultValue = this.embeddings.get(defaultKey);
    }

    public double[] vectorFor(T value) {
        return embeddings.getOrDefault(keyExtractor.apply(value), defaultValue);
    }

    public int size() {
        return defaultValue.length;
    }

    public static <T, K> Embedding<T, K> loadEmbedding(File file, Function<String, K> keyParser,
                                                       Function<T, K> keyExtractor, K defaultKey) {
        try {
            final Map<K, double[]> embeddings = new HashMap<>();
            try (Scanner scanner = new Scanner(file)) {
                scanner.useLocale(Locale.US);
                final int size = scanner.nextInt();
                while (scanner.hasNext()) {
                    final K key = keyParser.apply(scanner.next());
                    final double[] vector = new double[size];
                    for (int i = 0; i < size; i++) {
                        vector[i] = scanner.nextDouble();
                    }
                    embeddings.put(key, vector);
                }
            }
            return new Embedding<>(embeddings, keyExtractor, defaultKey);
        } catch (Exception e) {
            return null;
        }
    }

    public double[] defaultVector() {
        return defaultValue;
    }
}

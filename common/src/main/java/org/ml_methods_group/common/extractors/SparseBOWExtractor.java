package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SparseBOWExtractor<T> implements FeaturesExtractor<List<T>, SparseBOWExtractor.SparseBOWVector> {

    private final HashMap<String, Integer> indexes;
    private final List<HashExtractor<T>> hashers;

    public SparseBOWExtractor(HashMap<String, Integer> indexes, List<HashExtractor<T>> hashers) {
        this.indexes = indexes;
        this.hashers = hashers;
    }

    @Override
    public SparseBOWVector process(List<T> values) {
        final HashMap<Integer, Integer> counter = new HashMap<>();
        for (T value : values) {
            for (HashExtractor<T> hasher : hashers) {
                int index = indexes.getOrDefault(hasher.process(value), -1);
                if (index != -1) {
                    counter.put(index, counter.getOrDefault(index, 0) + 1);
                }
            }
        }
        return new SparseBOWVector(counter);
    }

    public static <T> HashMap<String, Integer> getDictionary(List<HashExtractor<T>> hashers, List<T> values) {
        final HashMap<String, Integer> counters = new HashMap<>();
        for (T value : values) {
            for (HashExtractor<T> hasher : hashers) {
                final String hash = hasher.process(value);
                counters.put(hash, counters.getOrDefault(hash, 0) + 1);
            }
        }
        final HashMap<String, Integer> indexes = new HashMap<>();
        counters.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .forEachOrdered(hash -> indexes.put(hash, indexes.size()));
        return indexes;
    }

    public static class SparseBOWVector {

        private final Map<Integer, Integer> counterByIndex;

        private final double norm;

        public SparseBOWVector(Map<Integer, Integer> counterByIndex) {
            final int squaredSum = counterByIndex.values().stream().mapToInt(x -> x * x).sum();
            this.counterByIndex = counterByIndex;
            this.norm = Math.sqrt(squaredSum);
        }

    }

    public static double cosineDistance(SparseBOWExtractor.SparseBOWVector a, SparseBOWExtractor.SparseBOWVector b) {
        int p = FunctionsUtils.scalarProduct(a.counterByIndex, b.counterByIndex);
        return p == 0 || a.norm == 0 || b.norm == 0 ? 1 : (1 - p / (a.norm * b.norm)) / 2;
    }
}

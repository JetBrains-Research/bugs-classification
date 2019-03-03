package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class BOWExtractor<T> implements FeaturesExtractor<List<T>, BOWExtractor.BOWVector> {

    private final HashMap<String, Integer> indexes;
    private final List<HashExtractor<T>> hashers;

    public BOWExtractor(HashMap<String, Integer> indexes, List<HashExtractor<T>> hashers) {
        this.indexes = indexes;
        this.hashers = hashers;
    }

    @Override
    public BOWVector process(List<T> values) {
        final int[] result = new int[indexes.size()];
        for (T value : values) {
            for (HashExtractor<T> hasher : hashers) {
                int index = indexes.getOrDefault(hasher.process(value), -1);
                if (index != -1) {
                    result[index] += 1;
                }
            }
        }
        return new BOWVector(result, values.size() * hashers.size());
    }

    public static <T> HashMap<String, Integer> mostCommon(List<HashExtractor<T>> hashers, List<T> values, int n) {
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
                .limit(n)
                .map(Map.Entry::getKey)
                .forEachOrdered(hash -> indexes.put(hash, indexes.size()));
        System.out.println(counters.size());
        return indexes;
    }

    public static class BOWVector {
        private final int[] counters;
        private final double norm;
        private final double normApproximation;

        private BOWVector(int[] counters, int wordsCount) {
            final int sum = IntStream.of(counters).sum();
            final int squaredSum = IntStream.of(counters)
                    .map(x -> x * x)
                    .sum();
            this.counters = counters;
            this.norm = Math.sqrt(squaredSum);
            this.normApproximation = Math.sqrt(squaredSum + (wordsCount - sum));
        }

    }

    public static double cosineDistance(BOWVector a, BOWVector b) {
        int p = FunctionsUtils.scalarProduct(a.counters, b.counters);
        return a.norm == 0 || b.norm == 0 ? 1 : (1 - p / (a.norm * b.norm)) / 2;
    }

    public static double smartCosineDistance(BOWVector a, BOWVector b) {
        int p = FunctionsUtils.scalarProduct(a.counters, b.counters);
        return a.normApproximation == 0 || b.normApproximation == 0 ?
                1 : (1 - p / (a.normApproximation * b.normApproximation)) / 2;
    }
}

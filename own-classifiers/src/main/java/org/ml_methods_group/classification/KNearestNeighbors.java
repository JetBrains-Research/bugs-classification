package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.DistanceFunction;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class KNearestNeighbors<T> implements Classifier<T> {
    private final int k;
    private final Map<T, Integer> samples = new ConcurrentHashMap<>();
    private final DistanceFunction<T> metric;

    public KNearestNeighbors(int k, DistanceFunction<T> metric) {
        this.k = k;
        this.metric = metric;
    }

    @Override
    public void train(Map<T, Integer> samples) {
        this.samples.clear();
        this.samples.putAll(samples);
    }

    @Override
    public int classify(T value) {
        return samples.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByKey(Comparator.comparingDouble(x -> metric.distance(value, x))))
                .limit(k)
                .collect(Collectors.groupingBy(Map.Entry::getValue, Collectors.counting()))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(-1);
    }
}

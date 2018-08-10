package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.DistanceFunction;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;

public class KNearestNeighbors<T, M> implements Classifier<T, M> {
    private final int k;
    private final Map<T, M> marks = new HashMap<>();
    private final List<T> samples = new ArrayList<>();
    private final DistanceFunction<T> metric;

    public KNearestNeighbors(int k, DistanceFunction<T> metric) {
        this.k = k;
        this.metric = metric;
    }

    @Override
    public void train(Map<T, M> marks) {
        this.marks.clear();
        this.marks.putAll(marks);
        samples.clear();
        samples.addAll(this.marks.keySet());
    }

    @Override
    public Map<M, Double> reliability(T value) {
        return Utils.kNearest(value, samples, k, metric)
                .stream()
                .collect(Collectors.groupingBy(marks::get, Collectors.counting()))
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> (double) entry.getValue() / k));
    }
}

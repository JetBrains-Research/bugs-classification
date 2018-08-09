package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.DistanceFunction;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class NearestCluster<T> implements Classifier<T> {
    private final Map<Integer, List<T>> clusters = new ConcurrentHashMap<>();
    private final DistanceFunction<T> metric;

    public NearestCluster(DistanceFunction<T> metric) {
        this.metric = metric;
    }

    @Override
    public void train(Map<T, Integer> samples) {
        final Map<Integer, List<T>> buffer = samples.keySet()
                .stream()
                .collect(Collectors.groupingBy(samples::get));
        clusters.clear();
        clusters.putAll(buffer);

    }

    @Override
    public int classify(T value) {
        return clusters.keySet()
                .stream()
                .min(Comparator.comparingDouble(i -> clusters.get(i).stream()
                        .mapToDouble(e -> metric.distance(value, e))
                        .average()
                        .orElse(0)))
                .orElse(-1);
    }
}

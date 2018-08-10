package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.DistanceFunction;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class NearestCluster<V, M> implements Classifier<V, M> {
    private final Map<M, List<V>> clusters = new ConcurrentHashMap<>();
    private final DistanceFunction<V> metric;

    public NearestCluster(DistanceFunction<V> metric) {
        this.metric = metric;
    }

    @Override
    public void train(Map<V, M> samples) {
        final Map<M, List<V>> buffer = samples.keySet()
                .stream()
                .collect(Collectors.groupingBy(samples::get));
        clusters.clear();
        clusters.putAll(buffer);

    }

    @Override
    public M classify(V value) {
        return clusters.keySet()
                .stream()
                .min(Comparator.comparingDouble(i -> clusters.get(i).stream()
                        .mapToDouble(e -> metric.distance(value, e))
                        .average()
                        .orElse(0)))
                .orElse(null);
    }

    @Override
    public Map<M, Double> reliability(V value) {
        return clusters.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> estimateReliability(value, e.getValue())));
    }

    private double estimateReliability(V value, List<V> cluster) {
        return metric.upperBound() - cluster.stream()
                .mapToDouble(element -> metric.distance(value, element))
                .average()
                .orElseGet(metric::upperBound);
    }
}

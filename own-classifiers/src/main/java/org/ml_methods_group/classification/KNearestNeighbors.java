package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.Cluster;
import org.ml_methods_group.core.DistanceFunction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
    public void train(Map<Cluster<T>, M> train) {
        marks.clear();
        for (Entry<Cluster<T>, M> entry : train.entrySet()) {
            entry.getKey().forEach(x -> marks.put(x, entry.getValue()));
        }
        samples.clear();
        samples.addAll(marks.keySet());
    }

    @Override
    public Map<M, Double> reliability(T value) {
        return Utils.kNearest(value, samples, k, metric)
                .stream()
                .collect(Collectors.toMap(marks::get, x -> metric.upperBound() - metric.distance(value, x),
                        Double::sum))
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Entry::getKey, entry -> entry.getValue() / k));
    }
}

package org.ml_methods_group.classification.classifiers;

import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.MarkedClusters;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class NearestCluster<V, M> implements Classifier<V, M> {
    private final Map<Cluster<V>, M> clusters = new HashMap<>();
    private final DistanceFunction<V> metric;
    private final boolean mergeClusters;

    public NearestCluster(DistanceFunction<V> metric) {
        this(metric, false);
    }

    public NearestCluster(DistanceFunction<V> metric, boolean mergeClusters) {
        this.metric = metric;
        this.mergeClusters = mergeClusters;
    }

    @Override
    public void train(MarkedClusters<V, M> samples) {
        clusters.clear();
        if (!mergeClusters) {
            clusters.putAll(samples.getMarks());
        } else {
            samples.getMarks().entrySet().stream()
                    .collect(Collectors.toMap(Entry::getValue, Entry::getKey, Cluster::merge))
                    .forEach((key, value) -> clusters.put(value, key));
        }
    }


    @Override
    public Map<M, Double> reliability(V value) {
        return clusters.entrySet()
                .stream()
                .collect(Collectors.toMap(Entry::getValue, e -> estimateReliability(value, e.getKey()),
                        Math::max));
    }

    private double estimateReliability(V value, Cluster<V> cluster) {
        return 1 - 2 / Math.PI * Math.atan(cluster.stream()
                .mapToDouble(element -> metric.distance(value, element))
                .average()
                .orElseThrow(RuntimeException::new));
    }
}

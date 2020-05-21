package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.common.*;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class OptimizedOptionSelector<V> implements ManyOptionsSelector<V, V> {

    private final DistanceFunction<V> metric;
    private final int topClusters;
    private final Map<V, Cluster<V>> representatives;

    public OptimizedOptionSelector(Clusters<V> clusters, RepresentativesPicker<V> picker,
                                   DistanceFunction<V> metric, int topClusters) {
        this.metric = metric;
        this.topClusters = topClusters;
        this.representatives = clusters.getClusters().stream()
                .filter(cluster -> !cluster.getElements().isEmpty())
                .collect(Collectors.toMap(
                        cluster -> picker.getRepresentatives(cluster).get(0),
                        Function.identity(),
                        Cluster::merge)
                );
    }

    @Override
    public Optional<List<V>> selectOptions(V value) {
        final var queue = new PriorityQueue<V>(Comparator.comparing(option -> -metric.distance(option, value)));
        for (var representative : representatives.keySet()) {
            queue.offer(representative);
            if (queue.size() > topClusters) {
                queue.poll();
            }
        }
        final List<V> options = queue.stream()
                .map(representatives::get)
                .map(Cluster::getElements)
                .flatMap(Collection::stream)
                .collect(Collectors.toList());
        V closest = options.get(0);
        double minDistance = metric.distance(value, closest);
        for (V option : options.subList(1, options.size())) {
            final double distance = metric.distance(value, option, minDistance);
            if (distance < minDistance) {
                minDistance = distance;
                closest = option;
            }
        }
        return Optional.of(List.of(closest));
    }

    @Override
    public List<V> getAllPossibleOptions() {
        return representatives.values().stream()
                .map(Cluster::getElements)
                .flatMap(Collection::stream)
                .collect(Collectors.toList());
    }

    @Override
    public int getSelectionSize() { return 1; }
}

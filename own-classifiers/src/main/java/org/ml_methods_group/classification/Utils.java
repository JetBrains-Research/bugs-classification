package org.ml_methods_group.classification;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.Wrapper;

import java.util.List;
import java.util.TreeSet;
import java.util.stream.Collectors;

class Utils {
    static <V> List<V> kNearest(V value, List<V> targets, int k, DistanceFunction<V> metric) {
        double bound = Double.POSITIVE_INFINITY;
        final TreeSet<Wrapper<Double, Integer>> heap = new TreeSet<>(Wrapper::compare);
        for (int i = 0; i < targets.size(); i++) {
            final double distance = metric.distance(value, targets.get(i), bound);
            if (heap.size() < k || distance < bound) {
                heap.add(new Wrapper<>(distance, i));
                if (heap.size() > k) {
                    heap.pollLast();
                }
                bound = heap.last().getFeatures();
            }
        }
        return heap.stream()
                .mapToInt(Wrapper::getMeta)
                .mapToObj(targets::get)
                .collect(Collectors.toList());
    }
}

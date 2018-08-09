package org.ml_methods_group.core.basic.metrics;

import org.ml_methods_group.core.DistanceFunction;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ListDistanceFunction<T> implements DistanceFunction<List<T>> {
    @Override
    public double distance(List<T> first, List<T> second) {
        if (first.isEmpty() && second.isEmpty()) {
            return 0;
        }
        final Map<T, Long> firstCounters = first.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        final Map<T, Long> secondCounters = second.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        int intersection = 0;
        for (Map.Entry<T, Long> entry : firstCounters.entrySet()) {
            intersection += Math.min(entry.getValue(), secondCounters.getOrDefault(entry.getKey(), 0L));
        }
        return 1 - (double) intersection / (first.size() + second.size() - intersection);
    }
}

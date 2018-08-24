package org.ml_methods_group.core.basic.metrics;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.algorithms.AssignmentProblem;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.util.Collections.emptyList;

public class FuzzyListDistanceFunction<T> implements DistanceFunction<List<T>> {

    private final DistanceFunction<T> metric;
    private final Function<T, Integer> classifier;

    public FuzzyListDistanceFunction(DistanceFunction<T> metric, Function<T, Integer> classifier) {
        this.metric = metric;
        this.classifier = classifier;
    }

    @Override
    public double distance(List<T> first, List<T> second) {
        final Map<Integer, List<T>> firstGroups = first.stream()
                .collect(Collectors.groupingBy(classifier, Collectors.toList()));
        final Map<Integer, List<T>> secondGroups = second.stream()
                .collect(Collectors.groupingBy(classifier, Collectors.toList()));
        final double intersection = firstGroups.entrySet().stream()
                .mapToDouble(entry -> match(entry.getValue(), secondGroups.getOrDefault(entry.getKey(), emptyList())))
                .sum();
        return 1 - intersection / Math.max(first.size(), second.size());
    }

    private double match(List<T> first, List<T> second) {
        if (first.size() > second.size()) {
            return match(second, first);
        }
        if (first.isEmpty()) {
            return 0;
        }
        if (first.size() == 1) {
            return bestMatch(first.get(0), second);
        }
        final int[][] weights = new int[first.size()][second.size()];
        for (int i = 0; i < first.size(); i++) {
            for (int j = 0; j < second.size(); j++) {
                weights[i][j] = toDiscrete(metric.upperBound() - metric.distance(first.get(i), second.get(j)));
            }
        }
        return toFloat(new AssignmentProblem(weights).solve());
    }

    private double bestMatch(T element, List<T> list) {
        double best = Double.POSITIVE_INFINITY;
        for (T another : list) {
            final double current = metric.distance(element, another, best);
            if (current < best) {
                best = current;
            }
        }
        return toFloat(toDiscrete(metric.upperBound() - best));
    }
    
    private int toDiscrete(double value) {
        return (int) (value * 1000);
    }

    private double toFloat(int value) {
        return value / 1000.0;
    }
}

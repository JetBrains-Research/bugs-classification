package org.ml_methods_group.common.metrics.functions;

import org.ml_methods_group.common.SimilarityMetric;
import org.ml_methods_group.common.metrics.algorithms.AssignmentProblem;
import org.ml_methods_group.common.DistanceFunction;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.util.Collections.emptyList;

public class FuzzyJaccardDistanceFunction<T> implements DistanceFunction<List<T>> {

    private final SimilarityMetric<T> metric;

    public FuzzyJaccardDistanceFunction(SimilarityMetric<T> metric) {
        this.metric = metric;
    }

    @Override
    public double distance(List<T> first, List<T> second) {
        final Map<Integer, List<T>> firstGroups = first.stream()
                .collect(Collectors.groupingBy(metric::getElementType, Collectors.toList()));
        final Map<Integer, List<T>> secondGroups = second.stream()
                .collect(Collectors.groupingBy(metric::getElementType, Collectors.toList()));
        final double intersection = firstGroups.entrySet().stream()
                .mapToDouble(entry -> match(entry.getValue(), secondGroups.getOrDefault(entry.getKey(), emptyList())))
                .sum();
        return 1 - intersection / (first.size() + second.size() - intersection);
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
                weights[i][j] = toDiscrete(metric.measure(first.get(i), second.get(j)));
            }
        }
        return toFloat(new AssignmentProblem(weights, false).solve());
    }

    private double bestMatch(T element, List<T> list) {
        double best = 0;
        for (T another : list) {
            final double current = metric.measure(element, another);
            if (current > best) {
                best = current;
            }
        }
        return toFloat(toDiscrete(best));
    }
    
    private int toDiscrete(double value) {
        return (int) (value * 1000);
    }

    private double toFloat(int value) {
        return value / 1000.0;
    }
}

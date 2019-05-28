package org.ml_methods_group.testing.markers;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.SolutionMarksHolder;
import org.ml_methods_group.marking.markers.Marker;

import java.util.*;
import java.util.stream.Collectors;

public class MarksHolderBasedMarker implements Marker<Cluster<Solution>, String> {

    private final SolutionMarksHolderExpander oracle;
    private final SolutionMarksHolder holder;
    private final double rate;
    private final int threshold;

    public MarksHolderBasedMarker(SolutionMarksHolder holder, double rate, int threshold,
                                  SolutionMarksHolderExpander oracle) {
        this.holder = holder;
        this.rate = rate;
        this.threshold = threshold;
        this.oracle = oracle;
    }

    public MarksHolderBasedMarker(SolutionMarksHolder holder, double rate, int threshold) {
        this(holder, rate, threshold, null);
    }

    public MarksHolderBasedMarker(SolutionMarksHolder holder) {
        this(holder, 0.8, 3);
    }

    @Override
    public String mark(Cluster<Solution> value) {
        if (value.size() < threshold) {
            return null;
        }
        HashMap<String, Integer> counters = new HashMap<>();
        int count = 0;
        for (Solution solution : value) {
            final Optional<List<String>> marks = holder.getMarks(solution);
            if (marks.isPresent()) {
                count++;
                marks.get().forEach(x -> counters.put(x, 1 + counters.getOrDefault(x, 0)));
            }
        }
        if (count < threshold && oracle != null) {
            expandMarks(value, counters, threshold - count);
            count = threshold;
        }
        final Optional<Map.Entry<String, Integer>> res =
                counters.entrySet().stream().max(Map.Entry.comparingByValue());
        final String data = counters.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(10)
                .map(e -> e.getKey() + "[" + e.getValue() + "]")
                .collect(Collectors.joining(" "));
        if (count < threshold || res.isEmpty() || res.get().getValue() < rate * count) {
            System.out.println("Verdict " + value.size() + " " + count + " " + data + " -> null");
            return null;
        }
        System.out.println("Verdict " + value.size() + " " + count + " " + data + " -> " + res.get().getKey());
        return res.get().getKey();
    }

    private void expandMarks(Cluster<Solution> cluster, Map<String, Integer> counters, int k) {
        final List<Solution> solutions = cluster.getElements();
        final Random random = new Random();
        while (k != 0) {
            final Solution solution = solutions.get(random.nextInt(solutions.size()));
            if (holder.getMarks(solution).isEmpty()) {
                for (String mark : oracle.expand(solution, holder)) {
                    counters.put(mark, counters.getOrDefault(mark, 0) + 1);
                }
                k--;
            }
        }
    }
}

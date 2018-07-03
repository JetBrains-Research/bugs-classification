package org.ml_methods_group.classification;

import org.ml_methods_group.core.Classifier;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.Comparator;
import java.util.Map;
import java.util.stream.Collectors;

public class KNearestNeighbors implements Classifier<Wrapper> {
    private final int k;
    private Map<Wrapper, Integer> samples;

    public KNearestNeighbors(int k) {
        this.k = k;
    }

    @Override
    public void train(Map<Wrapper, Integer> samples) {
        this.samples = samples;
    }

    @Override
    public int classify(Wrapper value) {
        return samples.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByKey(Comparator.comparingDouble(value::euclideanDistance)))
                .limit(k)
                .collect(Collectors.groupingBy(Map.Entry::getValue, Collectors.counting()))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(0);
    }
}

package org.ml_methods_group.testing;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiPredicate;
import java.util.stream.IntStream;

public class ClassificationTestingResult implements TestingResults {
    private final List<Double> confidence = new ArrayList<>();
    private final List<Boolean> isAcceptable = new ArrayList<>();

    void addTestResult(double confidence, boolean isAcceptable) {
        this.confidence.add(confidence);
        this.isAcceptable.add(isAcceptable);
    }

    private long count(BiPredicate<Double, Boolean> predicate) {
        return IntStream.range(0, confidence.size())
                .filter(i -> predicate.test(confidence.get(i), isAcceptable.get(i)))
                .count();
    }

    public double getCoverage(double threshold) {
        return (double) count((x, a) -> x >= threshold && a) / confidence.size();
    }

    public double getPrecision(double threshold) {
        return (double) count((x, a) -> x >= threshold && a) / count((x, a) -> x >= threshold);
    }

    public double getAccuracy() {
        return (double) count((x, a) -> a) / confidence.size();
    }

    @Override
    public double getValue() {
        return getAccuracy();
    }
}

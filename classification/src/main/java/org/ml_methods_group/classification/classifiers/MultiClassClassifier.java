package org.ml_methods_group.classification.classifiers;

import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.MarkedClusters;
import org.ml_methods_group.common.Wrapper;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class MultiClassClassifier<F, T, M> implements Classifier<Wrapper<F, T>, M> {

    private final Supplier<Classifier<Wrapper<F, T>, Boolean>> binaryClassifiers;
    private final Map<M, Classifier<Wrapper<F, T>, Boolean>> classifiers = new HashMap<>();
    private final M trashMark;

    public MultiClassClassifier(Supplier<Classifier<Wrapper<F, T>, Boolean>> binaryClassifiers, M trashMark) {
        this.binaryClassifiers = binaryClassifiers;
        this.trashMark = trashMark;
    }

    @Override
    public void train(MarkedClusters<Wrapper<F, T>, M> samples) {
        classifiers.clear();
        final Map<Wrapper<F, T>, M> dataset = samples.getFlatMarks();
        dataset.values().stream()
                .filter(x -> !Objects.equals(x, trashMark))
                .distinct()
                .forEachOrdered(mark -> {
                    final Map<Cluster<Wrapper<F, T>>, Boolean> view = samples.getMarks().entrySet().stream()
                            .collect(Collectors.toMap(Map.Entry::getKey, e -> Objects.equals(mark, e.getValue())));
                    final Classifier<Wrapper<F, T>, Boolean> binaryClassifier = binaryClassifiers.get();
                    binaryClassifier.train(new MarkedClusters<>(view));
                    classifiers.put(mark, binaryClassifier);
                });
    }

    @Override
    public Map<M, Double> reliability(Wrapper<F, T> value) {
        return classifiers.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey,
                        e -> e.getValue().reliability(value).get(true)));
    }
}

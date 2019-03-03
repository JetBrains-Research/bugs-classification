package org.ml_methods_group.common;

import java.io.Serializable;
import java.util.Collections;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;

public interface Classifier<T, M> extends Serializable {
    void train(MarkedClusters<T, M> samples);

    default Optional<M> classify(T value) {
        return reliability(value)
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey);
    }

    Map<M, Double> reliability(T value);

    default Map.Entry<M, Double> mostProbable(T value) {
        return reliability(value).entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow(NoSuchElementException::new);
    }
}

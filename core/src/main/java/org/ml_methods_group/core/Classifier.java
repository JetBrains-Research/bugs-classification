package org.ml_methods_group.core;

import java.util.Map;

public interface Classifier<T> {
    void train(Map<T, Integer> samples);
    int classify(T value);
}

package org.ml_methods_group.core;

import java.io.Serializable;
import java.util.Map;

public interface Classifier<T> extends Serializable {
    void train(Map<T, Integer> samples);
    int classify(T value);
}

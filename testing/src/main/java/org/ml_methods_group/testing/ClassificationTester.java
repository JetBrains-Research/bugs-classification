package org.ml_methods_group.testing;


import org.ml_methods_group.common.Classifier;

public interface ClassificationTester<V, M> {
    ClassificationTestingResult test(Classifier<V, M> classifier);
}

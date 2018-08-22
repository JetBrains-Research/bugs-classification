package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.Classifier;

public interface ClassificationTester<V, M> {
    ClassificationTestingResult test(Classifier<V, M> classifier);
}

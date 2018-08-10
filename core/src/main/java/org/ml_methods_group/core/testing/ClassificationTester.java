package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.Classifier;

public interface ClassificationTester<V, M> {
    ClassificationTestingResult<V, M> test(Classifier<V, M> classifier);
}

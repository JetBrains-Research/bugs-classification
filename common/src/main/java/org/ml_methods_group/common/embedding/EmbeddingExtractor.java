package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.FeaturesExtractor;

public interface EmbeddingExtractor<T> extends FeaturesExtractor<T, double[]> {
    double[] defaultVector();
}

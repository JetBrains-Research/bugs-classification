package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

public interface EmbeddingExtractor<T> extends FeaturesExtractor<T, double[]> {
    double[] defaultVector();
}

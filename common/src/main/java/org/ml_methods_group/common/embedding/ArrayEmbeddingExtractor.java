package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

import java.util.Arrays;

public class ArrayEmbeddingExtractor<T> implements EmbeddingExtractor<T[]>{

    private final EmbeddingExtractor<T> extractor;

    public ArrayEmbeddingExtractor(EmbeddingExtractor<T> extractor) {
        this.extractor = extractor;
    }

    @Override
    public double[] process(T[] value) {
        return Arrays.stream(value)
                .map(extractor::process)
                .reduce(FunctionsUtils::sum)
                .orElse(extractor.defaultVector());
    }

    @Override
    public double[] defaultVector() {
        return extractor.defaultVector();
    }
}

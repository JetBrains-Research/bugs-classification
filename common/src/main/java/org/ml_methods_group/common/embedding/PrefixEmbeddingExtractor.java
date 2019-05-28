package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

public class PrefixEmbeddingExtractor<T> implements EmbeddingExtractor<T[]> {

    private final EmbeddingExtractor<T> extractor;
    private final double[] noneVector;
    private final int elementsLimit;

    public PrefixEmbeddingExtractor(EmbeddingExtractor<T> extractor, int elementsLimit) {
        this.extractor = extractor;
        this.noneVector = extractor.defaultVector();
        this.elementsLimit = elementsLimit;
    }

    @Override
    public double[] process(T[] value) {
        final double[] result = new double[noneVector.length];
        for (int i = 0; i < elementsLimit; i++) {
            if (i < value.length) {
                FunctionsUtils.add(result, extractor.process(value[i]));
            } else {
                FunctionsUtils.add(result, noneVector);
            }
        }
        return result;
    }

    @Override
    public double[] defaultVector() {
        return FunctionsUtils.scale(noneVector, elementsLimit);
    }
}

package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

public class SequenceContextEmbeddingExtractor<T> implements
        EmbeddingExtractor<SequenceContextEmbeddingExtractor.SequenceContext<T>> {

    private final EmbeddingExtractor<T> leftExtractor;
    private final double[] leftNoneVector;
    private final EmbeddingExtractor<T> rightExtractor;
    private final double[] rightNoneVector;
    private final int windowSize;

    public SequenceContextEmbeddingExtractor(EmbeddingExtractor<T> leftExtractor,
                                             EmbeddingExtractor<T> rightExtractor,
                                             int windowSize) {
        this.leftExtractor = leftExtractor;
        this.leftNoneVector = leftExtractor.defaultVector();
        this.rightExtractor = rightExtractor;
        this.rightNoneVector = rightExtractor.defaultVector();
        this.windowSize = windowSize;
    }


    @Override
    public double[] process(SequenceContext<T> value) {
        final double[] result = new double[leftNoneVector.length];
        for (int shift = 1; shift <= windowSize; shift++) {
            int i = value.index - shift;
            if (0 <= i && i < value.sequence.length) {
                FunctionsUtils.add(result, leftExtractor.process(value.sequence[i]));
            } else {
                FunctionsUtils.add(result, leftNoneVector);
            }
        }
        for (int shift = 1; shift <= windowSize; shift++) {
            int i = value.index + shift;
            if (0 <= i && i < value.sequence.length) {
                FunctionsUtils.add(result, leftExtractor.process(value.sequence[i]));
            } else {
                FunctionsUtils.add(result, leftNoneVector);
            }
        }
        return result;
    }

    @Override
    public double[] defaultVector() {
        return FunctionsUtils.sum(FunctionsUtils.scale(leftNoneVector, windowSize),
                FunctionsUtils.scale(rightNoneVector, windowSize));
    }

    public static class SequenceContext<T> {
        private final T[] sequence;
        private final int index;

        public SequenceContext(T[] sequence, int index) {
            this.sequence = sequence;
            this.index = index;
        }
    }
}

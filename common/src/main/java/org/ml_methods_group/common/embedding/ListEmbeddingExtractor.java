package org.ml_methods_group.common.embedding;

import org.ml_methods_group.common.metrics.functions.FunctionsUtils;

import java.util.List;

public class ListEmbeddingExtractor<T> implements EmbeddingExtractor<List<T>> {

    public enum Reducing {SUM, MEAN}

    private final EmbeddingExtractor<T> extractor;
    private final Reducing reducing;

    public ListEmbeddingExtractor(EmbeddingExtractor<T> extractor, Reducing reducing) {
        this.extractor = extractor;
        this.reducing = reducing;
    }

//    @Override
//    public double[] process(List<T> value) {
//        final double[] result = value.stream()
//                .map(extractor::process)
//                .reduce(FunctionsUtils::sum)
//                .orElse(extractor.defaultVector());
//        for (int i = 0; i < result.length; i++) {
//            result[i] /= value.size();
//        }
//        return result;
//    }

    @Override
    public double[] process(List<T> value) {
        final double[] result = value.stream()
                .map(extractor::process)
                .reduce(FunctionsUtils::sum)
                .orElse(extractor.defaultVector());
        if (reducing == Reducing.MEAN) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= value.size();
            }
        }
        return result;
    }

    @Override
    public double[] defaultVector() {
        return extractor.defaultVector();
    }
}
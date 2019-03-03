package org.ml_methods_group.common.embedding;

import java.util.List;

public class NormalizedVectorExtractor<T> implements EmbeddingExtractor<T> {

    private final EmbeddingExtractor<T> extractor;
    private final double[] mean;
    private final double[] std;

    public NormalizedVectorExtractor(EmbeddingExtractor<T> extractor, double[] mean, double[] std) {
        this.extractor = extractor;
        this.mean = mean;
        this.std = std;
    }

    @Override
    public double[] process(T value) {
        return normalized(extractor.process(value));
    }

    private double[] normalized(double[] a) {
        final double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = (a[i] - mean[i]) / std[i];
        }
        return result;
    }

    public static <T> NormalizedVectorExtractor<T> normalization(List<T> values, EmbeddingExtractor<T> extractor) {
        final int n = extractor.defaultVector().length;
        final double[] mean = new double[n];
        final double[] meanSquare = new double[n];
        for (T value : values) {
            double[] vector = extractor.process(value);
            for (int i = 0; i < n; i++) {
                mean[i] += vector[i];
                meanSquare[i] += vector[i] * vector[i];
            }
        }
        final double[] std = new double[n];
        for (int i = 0; i < n; i++) {
            mean[i] /= values.size();
            meanSquare[i] /= values.size();
            std[i] = Math.sqrt(meanSquare[i] - mean[i] * mean[i]);
        }
        return new NormalizedVectorExtractor<>(extractor, mean, std);
    }

    @Override
    public double[] defaultVector() {
        return normalized(extractor.defaultVector());
    }
}

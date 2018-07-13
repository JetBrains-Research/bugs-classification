package org.ml_methods_group.core;

public interface AnalysisStrategy<V, F> {
    void analyze(V value);
    FeaturesExtractor<V, F> generateFeaturesExtractor();
}

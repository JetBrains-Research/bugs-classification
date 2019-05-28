package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.FeaturesExtractor;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class PointwiseExtractor<T, F> implements FeaturesExtractor<List<T>, List<F>> {

    private final FeaturesExtractor<T, F> extractor;

    public PointwiseExtractor(FeaturesExtractor<T, F> extractor) {
        this.extractor = extractor;
    }

    @Override
    public List<F> process(List<T> values) {
        return values.stream()
                .map(extractor::process)
                .collect(Collectors.toCollection(ArrayList::new));
    }
}

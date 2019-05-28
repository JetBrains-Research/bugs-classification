package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.FeaturesExtractor;

import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class FoldExtractor<V, F> implements FeaturesExtractor<V, F> {

    private final FeaturesExtractor<V, List<F>> extractor;
    private final BinaryOperator<F> combiner;
    private final Supplier<F> defaultValue;

    public FoldExtractor(FeaturesExtractor<V, List<F>> extractor, BinaryOperator<F> combiner,
                         Supplier<F> defaultValue) {
        this.extractor = extractor;
        this.combiner = combiner;
        this.defaultValue = defaultValue;
    }

    @Override
    public F process(V value) {
        final List<F> features = extractor.process(value);
        return features.stream()
                .reduce(defaultValue.get(), combiner);
    }
}

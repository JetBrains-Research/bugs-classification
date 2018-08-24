package org.ml_methods_group.core;

public interface BiFeaturesExtractor<V, O, F> {
    F process(V value, O option);

    default FeaturesExtractor<V, F> constOption(O option) {
        return x -> process(x, option);
    }
}

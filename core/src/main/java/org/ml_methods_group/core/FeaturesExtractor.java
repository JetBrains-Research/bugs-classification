package org.ml_methods_group.core;

import java.io.Serializable;

public interface FeaturesExtractor<V, F> extends Serializable {
    F process(V value);

    default <R> FeaturesExtractor<V, R> compose(FeaturesExtractor<? super F, R> other) {
        return x -> other.process(process(x));
    }
}

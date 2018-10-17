package org.ml_methods_group.common;

import java.io.Serializable;

public interface BiFeaturesExtractor<V, O, F> extends Serializable {
    F process(V value, O option);

    default FeaturesExtractor<V, F> constOption(O option) {
        return x -> process(x, option);
    }
}

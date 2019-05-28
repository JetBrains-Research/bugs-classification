package org.ml_methods_group.common;

import java.io.Serializable;

@FunctionalInterface
public interface SimilarityMetric<V> extends Serializable {
    double measure(V first, V second);

    /**
     * Similarity of elements, which are related to different groups, must be zero.
     * @return element group
     */
    default int getElementType(V value) {
        return 0;
    }
}

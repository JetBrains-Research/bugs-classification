package org.ml_methods_group.common;

@FunctionalInterface
public interface SimilarityMetric<V> {
    double measure(V first, V second);

    /**
     * Similarity of elements, which are related to different groups, must be zero.
     * @return element group
     */
    default int getElementType(V value) {
        return 0;
    }
}

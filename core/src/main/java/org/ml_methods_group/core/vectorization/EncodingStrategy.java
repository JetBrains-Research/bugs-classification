package org.ml_methods_group.core.vectorization;

public interface EncodingStrategy<T> {
    long encode(T value);
}

package org.ml_methods_group.core.vectorization;

import java.io.Serializable;

public interface EncodingStrategy<T> extends Serializable {
    long encode(T value);
}

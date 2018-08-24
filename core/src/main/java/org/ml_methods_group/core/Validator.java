package org.ml_methods_group.core;

public interface Validator<V, M> {
    boolean isValid(V value, M mark);
}

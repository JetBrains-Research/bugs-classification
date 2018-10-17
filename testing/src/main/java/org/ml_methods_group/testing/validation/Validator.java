package org.ml_methods_group.testing.validation;

public interface Validator<V, M> {
    boolean isValid(V value, M mark);
}

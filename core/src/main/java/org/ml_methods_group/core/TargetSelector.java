package org.ml_methods_group.core;

public interface TargetSelector<T> {
    void addTarget(T target);
    T selectTarget(T value);
}

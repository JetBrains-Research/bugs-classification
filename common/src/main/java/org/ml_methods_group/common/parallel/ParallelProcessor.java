package org.ml_methods_group.common.parallel;

@FunctionalInterface
public interface ParallelProcessor<V, C, A> {
    A process(V value, C context, A accumulator);
}

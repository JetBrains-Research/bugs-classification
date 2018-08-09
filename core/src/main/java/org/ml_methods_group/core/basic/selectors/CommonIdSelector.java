package org.ml_methods_group.core.basic.selectors;

import org.ml_methods_group.core.TargetSelector;

import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.ToIntFunction;

public class CommonIdSelector<T> implements TargetSelector<T> {

    private final Map<Integer, T> targets = new HashMap<>();
    private final ToIntFunction<T> idExtractor;

    public CommonIdSelector(ToIntFunction<T> idExtractor) {
        this.idExtractor = idExtractor;
    }

    @Override
    public void addTarget(T target) {
        targets.put(idExtractor.applyAsInt(target), target);
    }

    @Override
    public T selectTarget(T value) {
        return targets.get(idExtractor.applyAsInt(value));
    }
}

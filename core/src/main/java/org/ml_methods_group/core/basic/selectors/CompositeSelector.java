package org.ml_methods_group.core.basic.selectors;

import org.ml_methods_group.core.TargetSelector;

import java.util.Arrays;
import java.util.List;

public class CompositeSelector<T> implements TargetSelector<T> {

    private final List<TargetSelector<T>> selectors;

    @SafeVarargs
    public CompositeSelector(TargetSelector<T>... selectors) {
        this.selectors = Arrays.asList(selectors);
    }

    @Override
    public void addTarget(T target) {
        selectors.forEach(selector -> selector.addTarget(target));
    }

    @Override
    public T selectTarget(T value) {
        for (TargetSelector<T> selector : selectors) {
            final T target = selector.selectTarget(value);
            if (target != null) {
                return target;
            }
        }
        return null;
    }
}

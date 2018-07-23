package org.ml_methods_group.core.selection;

import org.ml_methods_group.core.Selector;

import java.util.*;

public class RandomSelector<T> implements Selector<T> {

    final Random random;

    public RandomSelector(long seed) {
        this.random = new Random(seed);
    }

    public RandomSelector() {
        this.random = new Random();
    }

    @Override
    public T getCenter(List<T> samples) {
        return samples.get(random.nextInt(samples.size()));
    }
}

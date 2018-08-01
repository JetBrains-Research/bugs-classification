package org.ml_methods_group.core.basic.selectors;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.TargetSelector;

import java.util.ArrayList;
import java.util.List;

public class ClosestTargetSelector<T> implements TargetSelector<T> {

    private final List<T> targets = new ArrayList<>();
    private final DistanceFunction<T> metric;

    public ClosestTargetSelector(DistanceFunction<T> metric) {
        this.metric = metric;
    }

    @Override
    public void addTarget(T target) {
        targets.add(target);
    }

    @Override
    public T selectTarget(T value) {
        double bestDistance = Double.POSITIVE_INFINITY;
        T bestTarget = null;
        for (T target : targets) {
            final double distance = metric.distance(value, target, bestDistance);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestTarget = target;
            }
        }
        return bestTarget;
    }
}

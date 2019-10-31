package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.OptionSelector;

import java.util.*;
import java.util.stream.Collectors;

public class KClosestPairsSelector<V> implements OptionSelector<V, List<V>> {

    private final List<V> options;
    private final DistanceFunction<V> metric;
    private final int k;

    public KClosestPairsSelector(List<V> options, DistanceFunction<V> metric, int k_nearest) {
        this.options = new ArrayList<>(options);
        this.metric = metric;
        this.k = k_nearest;
    }

    @Override
    public Optional<List<V>> selectOption(V value) {
        if (options.isEmpty()) {
            return Optional.empty();
        }
        ArrayList<V> kClosest = new ArrayList<>(Collections.nCopies(k, null));
        ArrayList<Double> kClosestDistances = new ArrayList<>(Collections.nCopies(k, Double.MAX_VALUE));
        kClosest.set(0, options.get(0));
        kClosestDistances.set(0, metric.distance(value, kClosest.get(0)));

        for (V option : options.subList(1, options.size())) {
            final double currentDistance = metric.distance(value, option, Double.MAX_VALUE);
            for (int i = 0; i < kClosestDistances.size(); ++i) {
                double kthDistance = kClosestDistances.get(i);
                if (kthDistance > currentDistance) {
                    shift(kClosestDistances, i);
                    shift(kClosest, i);
                    kClosestDistances.set(i, currentDistance);
                    kClosest.set(i, option);
                    break;
                }
            }
        }
        /*for (var dist : kClosestDistances)
            System.out.print(dist + " ");
        System.out.println();*/

        return Optional.of(kClosest);
    }

    private static <T> void shift(ArrayList<T> list, int from)
    {
        for(int i = list.size() - 1; i > from; i--)
        {
            list.set(i, list.get(i - 1));
        }
    }

    @Override
    public Collection<List<V>> getOptions() {
        return Collections.singleton(options);
    }
}

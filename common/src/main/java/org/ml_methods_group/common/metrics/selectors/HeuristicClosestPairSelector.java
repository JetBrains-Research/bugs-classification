package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.OptionSelector;

import java.util.*;
import java.util.stream.Collectors;

public class HeuristicClosestPairSelector<V, F, H> implements OptionSelector<V, V> {

    private final FeaturesExtractor<V, F> featuresExtractor;
    private final FeaturesExtractor<F, H> heuristicExtractor;
    private final DistanceFunction<F> metric;
    private final DistanceFunction<H> heuristicMetric;
    private final List<V> options;
    private final List<F> features;
    private final List<H> heuristics;

    public HeuristicClosestPairSelector(FeaturesExtractor<V, F> featuresExtractor,
                                        DistanceFunction<F> metric,
                                        FeaturesExtractor<F, H> heuristicExtractor,
                                        DistanceFunction<H> heuristicMetric,
                                        List<V> options) {
        this.featuresExtractor = featuresExtractor;
        this.heuristicExtractor = heuristicExtractor;
        this.metric = metric;
        this.heuristicMetric = heuristicMetric;
        this.options = new ArrayList<>(options);
        Collections.shuffle(this.options);
        this.features = this.options.stream()
                .map(featuresExtractor::process)
                .collect(Collectors.toCollection(ArrayList::new));
        this.heuristics = this.features.stream()
                .map(heuristicExtractor::process)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    @Override
    public Optional<V> selectOption(V value) {
        final F feature = featuresExtractor.process(value);
        final H heuristic = heuristicExtractor.process(feature);
        int best = -1;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int i = 0; i < options.size(); i++) {
            if (heuristicMetric.distance(heuristic, heuristics.get(i), bestDistance) >= bestDistance) {
                continue;
            }
            final double distance = metric.distance(feature, features.get(i), bestDistance);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        return best != -1 ? Optional.of(options.get(best)) : Optional.empty();
    }

    @Override
    public Collection<V> getOptions() {
        return Collections.unmodifiableList(options);
    }
}

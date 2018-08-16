package org.ml_methods_group.clusterization;

import org.ml_methods_group.core.Clusterer;
import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.Wrapper;
import org.ml_methods_group.core.parallel.ParallelContext;
import org.ml_methods_group.core.parallel.ParallelUtils;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class CompositeClusterer<V, F> implements Clusterer<V> {

    private final FeaturesExtractor<V, F> featuresExtractor;
    private final Clusterer<Wrapper<F, V>> clusterer;

    public CompositeClusterer(FeaturesExtractor<V, F> featuresExtractor, Clusterer<Wrapper<F, V>> clusterer) {
        this.featuresExtractor = featuresExtractor;
        this.clusterer = clusterer;
    }

    @Override
    public List<List<V>> buildClusters(List<V> values) {
        final Function<V, Wrapper<F, V>> processor = Wrapper.wrap(featuresExtractor::process);
        final List<Wrapper<F, V>> wrappers;
        try (ParallelContext context = new ParallelContext()) {
            wrappers = context.runParallelWithConsumer(
                    values,
                    ParallelUtils::defaultListImplementation,
                    (x, accumulator) -> accumulator.add(processor.apply(x)),
                    ParallelUtils::combineLists);
        }
        return clusterer.buildClusters(wrappers)
                .stream()
                .map(cluster -> cluster.stream()
                        .map(Wrapper::getMeta)
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }
}

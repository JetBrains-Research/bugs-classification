package org.ml_methods_group.clustering.clusterers;

import org.ml_methods_group.common.Clusterer;
import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Wrapper;
import org.ml_methods_group.common.parallel.ParallelContext;
import org.ml_methods_group.common.parallel.ParallelUtils;

import java.util.List;
import java.util.function.Function;

public class CompositeClusterer<V, F> implements Clusterer<V> {

    private final FeaturesExtractor<V, F> featuresExtractor;
    private final Clusterer<Wrapper<F, V>> clusterer;

    public CompositeClusterer(FeaturesExtractor<V, F> featuresExtractor, Clusterer<Wrapper<F, V>> clusterer) {
        this.featuresExtractor = featuresExtractor;
        this.clusterer = clusterer;
    }

    @Override
    public Clusters<V> buildClusters(List<V> values) {
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
                .map(Wrapper::getMeta);
    }
}

package org.ml_methods_group.core.basic.extractors;

import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.changes.CodeChange;
import org.ml_methods_group.core.parallel.ParallelContext;
import org.ml_methods_group.core.parallel.ParallelUtils;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.core.vectorization.VectorTemplate.Postprocessor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class VectorExtractor implements FeaturesExtractor<List<CodeChange>, double[]> {

    private volatile VectorTemplate template;

    public VectorExtractor(List<List<CodeChange>> dataset, int bound, Postprocessor postprocessor,
                           List<EncodingStrategy> strategies) {
        this.template = train(dataset, strategies, bound, postprocessor);
    }

    @Override
    public double[] process(List<CodeChange> changes) {
        return template.process(changes);
    }

    private static VectorTemplate train(List<List<CodeChange>> dataset, List<EncodingStrategy> strategies,
                                        int lowerBound, Postprocessor postprocessor) {
        try (ParallelContext context = new ParallelContext()) {
            final Map<Long, Integer> buffer =
                    context.runParallelWithConsumer(
                            dataset,
                            ParallelUtils::defaultMapImplementation,
                            (changes, accumulator) -> index(changes, strategies, accumulator),
                            ParallelUtils::combineCounters);
            final List<Long> acceptable = buffer.entrySet()
                    .stream()
                    .filter(entry -> entry.getValue() >= lowerBound)
                    .map(Entry::getKey)
                    .collect(Collectors.toList());
            return new VectorTemplate(acceptable, postprocessor, strategies);
        }
    }

    private static void index(List<CodeChange> changes, List<EncodingStrategy> strategies,
                              Map<Long, Integer> accumulator) {
        for (EncodingStrategy strategy : strategies) {
            for (CodeChange change : changes) {
                final long code = strategy.encode(change);
                accumulator.compute(code, (x, old) -> old == null ? 1 : 1 + old);
            }
        }
    }
}

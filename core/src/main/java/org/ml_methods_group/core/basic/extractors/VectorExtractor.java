package org.ml_methods_group.core.basic.extractors;

import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.parallel.ParallelContext;
import org.ml_methods_group.core.parallel.ParallelUtils;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.core.vectorization.VectorTemplate.Postprocessor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class VectorExtractor implements FeaturesExtractor<double[]> {

    private final ChangeGenerator generator;
    private final int bound;
    private final List<EncodingStrategy> strategies;
    private final Postprocessor postprocessor;
    private volatile VectorTemplate template;

    public VectorExtractor(ChangeGenerator generator, int bound, Postprocessor postprocessor,
                           List<EncodingStrategy> strategies) {
        this.generator = generator;
        this.bound = bound;
        this.strategies = strategies;
        this.postprocessor = postprocessor;
    }

    @Override
    public double[] process(Solution value, Solution target) {
        return template.process(generator.getChanges(value, target));
    }

    @Override
    public void train(Map<Solution, Solution> dataset) {
        try (ParallelContext context = new ParallelContext()) {
            final List<Entry<Solution, Solution>> pairs = new ArrayList<>(dataset.entrySet());
            final Map<Long, Integer> buffer = context.runParallel(
                    pairs,
                    HashMap::new,
                    (Entry<Solution, Solution> pair, Map<Long, Integer> accumulator) ->
                            index(pair.getKey(), pair.getValue(), accumulator),
                    ParallelUtils::combineCounters);
            final List<Long> acceptable = buffer.entrySet()
                    .stream()
                    .filter(entry -> entry.getValue() >= bound)
                    .map(Entry::getKey)
                    .collect(Collectors.toList());
            template = new VectorTemplate(acceptable, postprocessor, strategies);
        }
    }

    private void index(Solution solution, Solution target, Map<Long, Integer> accumulator) {
        final List<CodeChange> changes = generator.getChanges(solution, target);
        for (EncodingStrategy strategy : strategies) {
            for (CodeChange change : changes) {
                final long code = strategy.encode(change);
                accumulator.compute(code, (x, old) -> old == null ? 1 : 1 + old);
            }
        }
    }
}

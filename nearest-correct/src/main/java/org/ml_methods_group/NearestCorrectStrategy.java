package org.ml_methods_group;

import org.ml_methods_group.core.*;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.parallel.ParallelContext;
import org.ml_methods_group.core.parallel.ParallelUtils;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.core.vectorization.VectorTemplate;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.entities.Solution.Verdict.FAIL;
import static org.ml_methods_group.core.entities.Solution.Verdict.OK;

public class NearestCorrectStrategy implements AnalysisStrategy<Solution, double[]> {

    private final List<Solution> correctSolutions = new ArrayList<>();
    private final List<Solution> incorrectSolutions = new ArrayList<>();

    private final ChangeGenerator generator;
    private final List<EncodingStrategy> strategies;
    private final Supplier<List<Solution>> centersGenerator;
    private final DistanceFunction<Solution> metric;
    private final VectorTemplate.Postprocessor postprocessor;
    private FeaturesExtractor<Solution, double[]> extractor;

    public NearestCorrectStrategy(ChangeGenerator generator,
                                  Clusterer<Solution> clusterer,
                                  Selector<Solution> selector,
                                  DistanceFunction<Solution> metric,
                                  VectorTemplate.Postprocessor postprocessor,
                                  List<EncodingStrategy> strategies) {
        this.generator = generator;
        this.centersGenerator = () -> clusterer.buildClusters(correctSolutions).stream()
                .map(selector::getCenter)
                .collect(Collectors.toList());
        this.metric = metric;
        this.strategies = new ArrayList<>(strategies);
        this.postprocessor = postprocessor;
    }

    public NearestCorrectStrategy(ChangeGenerator generator,
                                  DistanceFunction<Solution> metric,
                                  VectorTemplate.Postprocessor postprocessor,
                                  List<EncodingStrategy> strategies) {
        this.generator = generator;
        this.centersGenerator = () -> correctSolutions;
        this.metric = metric;
        this.strategies = new ArrayList<>(strategies);
        this.postprocessor = postprocessor;
    }

    public NearestCorrectStrategy(ChangeGenerator generator, Clusterer<Solution> clusterer,
                                  Selector<Solution> selector,
                                  DistanceFunction<Solution> metric,
                                  VectorTemplate.Postprocessor postprocessor,
                                  EncodingStrategy... strategies) {
        this(generator, clusterer, selector, metric, postprocessor, Arrays.asList(strategies));
    }

    @Override
    public void analyze(Solution value) {
        if (value.getVerdict() == OK) {
            correctSolutions.add(value);
        } else if (value.getVerdict() == FAIL) {
            incorrectSolutions.add(value);
        }
        invalidate();
    }

    @Override
    public FeaturesExtractor<Solution, double[]> generateFeaturesExtractor() {
        if (extractor != null) {
            return extractor;
        }
        final List<Solution> centers = centersGenerator.get();
        System.out.println("Centers: " + centers.size());
        final Map<Integer, Solution> cache = new ConcurrentHashMap<>();
        final Map<Long, Integer> counters;
        try (ParallelContext context = new ParallelContext()) {
            long s = System.currentTimeMillis();
            counters = context.<Solution, Map<Long, Integer>>runParallel(incorrectSolutions, HashMap::new,
                    (solution, accumulator) -> index(solution, centers, cache, accumulator),
                    ParallelUtils::combineCounters);
            System.out.println("Time: " + (System.currentTimeMillis() - s));
        }
        final List<Long> codes = counters.entrySet().stream()
                .filter(entry -> entry.getValue() > 2)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        System.out.println("Codes: " + codes.size());
        final VectorTemplate template = new VectorTemplate(codes, postprocessor, strategies);
        return new NearestCorrectFeaturesExtractor(template, centers, cache, metric, generator);
    }

    private Map<Long, Integer> index(Solution solution, List<Solution> centers, Map<Integer, Solution> cache,
                                     Map<Long, Integer> counters) {
        final Solution target = findNearest(solution, centers, metric);
        cache.put(solution.getSolutionId(), target);
        final List<CodeChange> changes = generator.getChanges(solution, target);
        System.out.print("Changes: " + changes.size() + " " + (solution.getSessionId() == target.getSessionId()) + " ");
        System.out.println(solution.getSessionId() + " " + target.getSessionId());
        for (EncodingStrategy strategy : strategies) {
            for (CodeChange change : changes) {
                final long code = strategy.encode(change);
                counters.compute(code, (x, old) -> old == null ? 1 : 1 + old);
            }
        }
        return counters;
    }

    private void invalidate() {
        extractor = null;
    }

    private static Solution findNearest(Solution solution, List<Solution> targets, DistanceFunction<Solution> metric) {
        Solution best = targets.stream()
                .filter(target -> target.getSessionId() == solution.getSessionId())
                .findAny().orElse(targets.get(0));
        double currentMax = metric.distance(solution, best);
        for (Solution target : targets) {
            final double currentDistance = metric.distance(solution, target, currentMax);
            if (currentDistance < currentMax) {
                currentMax = currentDistance;
                best = target;
            }
        }
        return best;
    }

    private static class NearestCorrectFeaturesExtractor implements FeaturesExtractor<Solution, double[]> {

        private final VectorTemplate template;
        private final List<Solution> centers;
        private final Map<Integer, Solution> cache;
        private final DistanceFunction<Solution> metric;
        private final ChangeGenerator generator;

        private NearestCorrectFeaturesExtractor(VectorTemplate template, List<Solution> centers,
                                                Map<Integer, Solution> cache, DistanceFunction<Solution> metric,
                                                ChangeGenerator generator) {
            this.template = template;
            this.centers = centers;
            this.cache = cache;
            this.metric = metric;
            this.generator = generator;
        }

        @Override
        public double[] process(Solution value) {
            final Solution target = cache.computeIfAbsent(value.getSolutionId(),
                    x -> findNearest(value, centers, metric));
            final List<CodeChange> changes = generator.getChanges(value, target);
            return template.process(changes);
        }
    }
}

package org.ml_methods_group;

import org.ml_methods_group.core.*;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.core.vectorization.VectorTemplate;

import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.entities.Solution.Verdict.*;

public class NearestCorrectStrategy implements AnalysisStrategy<Solution, double[]> {

    private final Map<Long, Integer> codeCounters = new HashMap<>();

    private final List<Solution> correctSolutions = new ArrayList<>();
    private final List<Solution> incorrectSolutions = new ArrayList<>();

    private final ChangeGenerator generator;
    private final List<EncodingStrategy> strategies;
    private final Clusterer<Solution> clusterer;
    private final Selector<Solution> selector;
    private final DistanceFunction<Solution> metric;
    private final VectorTemplate.Postprocessor postprocessor;
    private FeaturesExtractor<Solution, double[]> extractor;

    public NearestCorrectStrategy(ChangeGenerator generator, Clusterer<Solution> clusterer,
                                  Selector<Solution> selector,
                                  DistanceFunction<Solution> metric,
                                  VectorTemplate.Postprocessor postprocessor,
                                  List<EncodingStrategy> strategies) {
        this.generator = generator;
        this.selector = selector;
        this.metric = metric;
        this.strategies = new ArrayList<>(strategies);
        this.clusterer = clusterer;
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
        clear();
        final List<Solution> centers =
//                clusterer.buildClusters(correctSolutions).stream()
//                .map(selector::getCenter)
//                .collect(Collectors.toList());
                correctSolutions;
        System.out.println("Centers: " + centers.size());
        incorrectSolutions.forEach(solution -> index(solution, centers));
        final List<Long> codes = codeCounters.entrySet().stream()
                .filter(entry -> entry.getValue() > 1)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
//        System.out.println("Codes: " + codes.size());
        final VectorTemplate template = new VectorTemplate(codes, postprocessor, strategies);
        return new NearestCorrectFeaturesExtractor(template, centers, metric, generator);
    }

    private void index(Solution solution, List<Solution> centers) {
        final Solution target = findNearest(solution, centers, metric);
        final List<CodeChange> changes = generator.getChanges(solution, target);
        System.out.println("Changes: " + changes.size() + " " + (solution.getSessionId() == target.getSessionId()));
        for (EncodingStrategy strategy : strategies) {
            for (CodeChange change : changes) {
                final long code = strategy.encode(change);
                codeCounters.compute(code, (x, old) -> old == null ? 1 : 1 + old);
            }
        }
    }

    private void invalidate() {
        extractor = null;
    }

    private void clear() {
        codeCounters.clear();
    }

    private static Solution findNearest(Solution solution, List<Solution> targets, DistanceFunction<Solution> metric) {
        double currentMax = Integer.MAX_VALUE;
        Solution best = null;
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
        private final DistanceFunction<Solution> metric;
        private final ChangeGenerator generator;

        private NearestCorrectFeaturesExtractor(VectorTemplate template, List<Solution> centers,
                                                DistanceFunction<Solution> metric, ChangeGenerator generator) {
            this.template = template;
            this.centers = centers;
            this.metric = metric;
            this.generator = generator;
        }

        @Override
        public double[] process(Solution value) {
            final Solution target = findNearest(value, centers, metric);
            final List<CodeChange> changes = generator.getChanges(value, target);
            return template.process(changes);
        }
    }
}

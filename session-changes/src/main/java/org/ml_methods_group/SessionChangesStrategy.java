package org.ml_methods_group;

import org.ml_methods_group.core.*;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.entities.CodeChange;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.core.vectorization.VectorTemplate;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.entities.Solution.Verdict.FAIL;
import static org.ml_methods_group.core.entities.Solution.Verdict.OK;

public class SessionChangesStrategy implements AnalysisStrategy<Solution, double[]> {

    private final Map<Long, Integer> codeCounters = new HashMap<>();

    private final Map<Integer, Solution> correctSolutions = new HashMap<>();
    private final List<Solution> incorrectSolutions = new ArrayList<>();

    private final ChangeGenerator generator;
    private final List<EncodingStrategy> strategies;
    private final VectorTemplate.Postprocessor postprocessor;
    private final DistanceFunction<Solution> metric;
    private FeaturesExtractor<Solution, double[]> extractor;

    public SessionChangesStrategy(ChangeGenerator generator,
                                  DistanceFunction<Solution> metric,
                                  VectorTemplate.Postprocessor postprocessor,
                                  List<EncodingStrategy> strategies) {
        this.generator = generator;
        this.strategies = new ArrayList<>(strategies);
        this.postprocessor = postprocessor;
        this.metric = metric;
    }

    public SessionChangesStrategy(ChangeGenerator generator,
                                  VectorTemplate.Postprocessor postprocessor,
                                  DistanceFunction<Solution> metric, EncodingStrategy... strategies) {
        this(generator, metric, postprocessor, Arrays.asList(strategies));
    }

    @Override
    public void analyze(Solution value) {
        if (value.getVerdict() == OK) {
            correctSolutions.put(value.getSessionId(), value);
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
        incorrectSolutions.forEach(solution -> index(solution, correctSolutions.get(solution.getSessionId())));
        final List<Long> codes = codeCounters.entrySet().stream()
                .filter(entry -> entry.getValue() > 1)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
//        System.out.println("Codes: " + codes.size());
        final VectorTemplate template = new VectorTemplate(codes, postprocessor, strategies);
        return new SessionChangesFeaturesExtractor(template, correctSolutions.values(), metric, generator);
    }

    private void index(Solution incorrect, Solution correct) {
        final List<CodeChange> changes = generator.getChanges(incorrect, correct);
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

    private static Solution findNearest(Solution solution, Collection<Solution> targets, DistanceFunction<Solution> metric) {
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

    private static class SessionChangesFeaturesExtractor implements FeaturesExtractor<Solution, double[]> {

        private final VectorTemplate template;
        private final Map<Integer, Solution> sessions;
        private final DistanceFunction<Solution> metric;
        private final ChangeGenerator generator;

        private SessionChangesFeaturesExtractor(VectorTemplate template, Collection<Solution> correct,
                                                DistanceFunction<Solution> metric, ChangeGenerator generator) {
            this.template = template;
            this.sessions = correct.stream()
                    .collect(Collectors.toMap(Solution::getSessionId, Function.identity()));
            this.metric = metric;
            this.generator = generator;
        }

        @Override
        public double[] process(Solution value) {
            final Solution target = sessions.containsKey(value.getSessionId()) ?
                    sessions.get(value.getSessionId()) : findNearest(value, sessions.values(), metric);
            final List<CodeChange> changes = generator.getChanges(value, target);
            return template.process(changes);
        }
    }
}

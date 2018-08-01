package org.ml_methods_group.core;

import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.parallel.ParallelContext;
import org.ml_methods_group.core.parallel.ParallelUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AnalysisStrategy<F> {

    private final List<Solution> incorrectSolutions = new ArrayList<>();
    private final TargetSelector<Solution> selector;
    private final FeaturesExtractor<F> extractor;

    public AnalysisStrategy(TargetSelector<Solution> selector,
                               FeaturesExtractor<F> extractor) {
        this.selector = selector;
        this.extractor = extractor;
    }

    public void analyze(Solution value) {
        switch (value.getVerdict()) {
            case FAIL:
                incorrectSolutions.add(value);
                break;
            case OK:
                selector.addTarget(value);
                break;
        }
    }

    public List<Wrapper<F, Solution>> generateFeatures(List<Solution> solutions) {
        try (ParallelContext context = new ParallelContext()) {
            final Map<Solution, Solution> train = context.runParallel(
                    incorrectSolutions,
                    HashMap::new,
                    (Solution solution, Map<Solution, Solution> accumulator) ->
                            accumulator.put(solution, selector.selectTarget(solution)),
                    ParallelUtils::combineMaps);

            extractor.train(train);
            return context.runParallel(
                    solutions,
                    ArrayList::new,
                    (Solution solution, List<Wrapper<F, Solution>> accumulator) ->
                            accumulator.add(new Wrapper<>(
                                    extractor.process(solution, selector.selectTarget(solution)),
                                    solution)),
                    ParallelUtils::combineLists);
        }
    }
}

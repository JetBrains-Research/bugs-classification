package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;

import java.util.Map;

public class ManyProblemsBasedChangesExtractor implements FeaturesExtractor<Solution, Changes> {

    Map<Integer, FeaturesExtractor<Solution, Changes>> generatorByProblemId;

    public ManyProblemsBasedChangesExtractor(Map<Integer, FeaturesExtractor<Solution, Changes>> generatorByProblemId) {
        this.generatorByProblemId = generatorByProblemId;
    }

    @Override
    public Changes process(Solution value) {
        FeaturesExtractor<Solution, Changes> oracle = generatorByProblemId.get(value.getProblemId());
        return oracle.process(value);
    }
}

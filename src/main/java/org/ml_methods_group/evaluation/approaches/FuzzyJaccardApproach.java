package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.metrics.functions.CodeChangeSimilarityMetric;
import org.ml_methods_group.common.metrics.functions.FuzzyJaccardDistanceFunction;

import java.io.Serializable;
import java.util.List;
import java.util.function.Function;

public class FuzzyJaccardApproach {

    public static final ApproachTemplate<Changes> TEMPLATE = (d, g) -> getDefaultApproach(g);

    public static Approach<Changes> getDefaultApproach(FeaturesExtractor<Solution, Changes> generator) {
        return new Approach<>(generator,
                CommonUtils.metricFor(
                        new FuzzyJaccardDistanceFunction<>(new CodeChangeSimilarityMetric()),
                        (Function<Changes, List<CodeChange>> & Serializable) Changes::getChanges), "fuz_jac");
    }
}

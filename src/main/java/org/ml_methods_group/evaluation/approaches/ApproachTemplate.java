package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;

public interface ApproachTemplate<T> {
    Approach<T> getApproach(Dataset dataset, FeaturesExtractor<Solution, Changes> generator);
}

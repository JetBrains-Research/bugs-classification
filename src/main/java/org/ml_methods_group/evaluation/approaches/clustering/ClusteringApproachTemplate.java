package org.ml_methods_group.evaluation.approaches.clustering;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.evaluation.approaches.Approach;
import org.ml_methods_group.evaluation.approaches.ApproachTemplate;

import java.util.function.BiFunction;

public class ClusteringApproachTemplate {

    private final BiFunction<Dataset, FeaturesExtractor<Solution, Changes>, ClusteringApproach> creator;

    public <T> ClusteringApproachTemplate(ApproachTemplate<T> template) {
        this.creator = (dataset, extractor) -> {
            final Approach<T> approach = template.getApproach(dataset, extractor);
            return new ClusteringApproach(approach.name, approach);
        };
    }


    public ClusteringApproach createApproach(Dataset train, FeaturesExtractor<Solution, Changes> generator) {
        return creator.apply(train, generator);
    }
}

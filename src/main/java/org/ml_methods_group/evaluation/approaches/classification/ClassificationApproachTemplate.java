package org.ml_methods_group.evaluation.approaches.classification;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.evaluation.approaches.Approach;
import org.ml_methods_group.evaluation.approaches.ApproachTemplate;

import java.util.function.BiFunction;

public class ClassificationApproachTemplate {

    private final BiFunction<Dataset, FeaturesExtractor<Solution, Changes>, ClassificationApproach> creator;

    public <T> ClassificationApproachTemplate(ApproachTemplate<T> template) {
        this.creator = (dataset, extractor) -> {
            final Approach<T> approach = template.getApproach(dataset, extractor);
            return new ClassificationApproach(approach);
        };
    }


    public ClassificationApproach createApproach(Dataset train, FeaturesExtractor<Solution, Changes> generator) {
        return creator.apply(train, generator);
    }
}

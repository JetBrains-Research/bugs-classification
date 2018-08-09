package org.ml_methods_group.core.basic.extractors;

import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.changes.CodeChange;
import org.ml_methods_group.core.entities.Solution;

import java.util.List;
import java.util.Map;

public class ChangeListExtractor implements FeaturesExtractor<List<CodeChange>> {

    private final ChangeGenerator generator;

    public ChangeListExtractor(ChangeGenerator generator) {
        this.generator = generator;
    }

    @Override
    public List<CodeChange> process(Solution value, Solution target) {
        return generator.getChanges(value, target);
    }

    @Override
    public void train(Map<Solution, Solution> dataset) {
    }
}

package org.ml_methods_group.core.basic.extractors;

import org.ml_methods_group.core.BiFeaturesExtractor;
import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.OptionSelector;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.changes.Changes;
import org.ml_methods_group.core.entities.Solution;

import java.util.NoSuchElementException;

public class ChangesExtractor implements FeaturesExtractor<Solution, Changes>,
        BiFeaturesExtractor<Solution, Solution, Changes> {

    private final ChangeGenerator generator;
    private final OptionSelector<Solution, Solution> selector;

    public ChangesExtractor(ChangeGenerator generator, OptionSelector<Solution, Solution> selector) {
        this.generator = generator;
        this.selector = selector;
    }

    @Override
    public Changes process(Solution value) {
        return selector.selectOption(value)
                .map(option -> generator.getChanges(value, option))
                .orElseThrow(NoSuchElementException::new);
    }

    @Override
    public Changes process(Solution value, Solution option) {
        return generator.getChanges(value, option);
    }
}

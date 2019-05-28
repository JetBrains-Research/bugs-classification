package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.BiFeaturesExtractor;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.OptionSelector;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.Solution;

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

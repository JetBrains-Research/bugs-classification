package org.ml_methods_group.core.basic.extractors;

import org.ml_methods_group.core.BiFeaturesExtractor;
import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.basic.selectors.OptionSelector;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.changes.CodeChange;
import org.ml_methods_group.core.entities.Solution;

import java.util.List;
import java.util.NoSuchElementException;

public class ChangeListExtractor implements FeaturesExtractor<Solution, List<CodeChange>>,
        BiFeaturesExtractor<Solution, Solution, List<CodeChange>> {

    private final ChangeGenerator generator;
    private final OptionSelector<Solution, Solution> selector;

    public ChangeListExtractor(ChangeGenerator generator, OptionSelector<Solution, Solution> selector) {
        this.generator = generator;
        this.selector = selector;
    }

    @Override
    public List<CodeChange> process(Solution value) {
        return selector.selectOption(value)
                .map(option -> generator.getChanges(value, option))
                .orElseThrow(NoSuchElementException::new);
    }

    @Override
    public List<CodeChange> process(Solution value, Solution option) {
        return generator.getChanges(value, option);
    }
}

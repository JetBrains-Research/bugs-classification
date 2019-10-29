package org.ml_methods_group.common.extractors;

import org.ml_methods_group.common.BiFeaturesExtractor;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.OptionSelector;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

public class KNearestNeighborsChangesExtractor implements FeaturesExtractor<Solution, List<Changes>>,
        BiFeaturesExtractor<Solution, List<Solution>, List<Changes>> {

    private final ChangeGenerator generator;
    private final OptionSelector<Solution, List<Solution>> selector;

    public KNearestNeighborsChangesExtractor(ChangeGenerator generator, OptionSelector<Solution, List<Solution>> selector) {
        this.generator = generator;
        this.selector = selector;
    }

    @Override
    public List<Changes> process(Solution value) {
        var options = selector.selectOption(value).orElseThrow(NoSuchElementException::new);
        return options.stream()
                .map(option -> generator.getChanges(value, option))
                .collect(Collectors.toList());
    }

    @Override
    public List<Changes> process(Solution value, List<Solution> options) {
        return options.stream()
                .map(option -> generator.getChanges(value, option))
                .collect(Collectors.toList());
    }
}
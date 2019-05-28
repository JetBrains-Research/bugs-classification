package org.ml_methods_group.common.ast.changes;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.Action;
import com.github.gumtreediff.matchers.CompositeMatchers.ClassicGumtree;
import com.github.gumtreediff.matchers.CompositeMatchers.CompleteGumtreeMatcher;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.generation.ASTGenerator;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

public class BasicChangeGenerator implements ChangeGenerator {
    private final ASTGenerator generator;
    private final List<BiFunction<ITree, ITree, Matcher>> factories;

    public BasicChangeGenerator(ASTGenerator generator, List<BiFunction<ITree, ITree, Matcher>> factories) {
        this.factories = factories;
        this.generator = generator;
    }

    public BasicChangeGenerator(ASTGenerator generator) {
        this(generator, Arrays.asList(
                (Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                        new CompleteGumtreeMatcher(x, y, new MappingStore()),
                (Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                        new ClassicGumtree(x, y, new MappingStore())
        ));
    }

    @Override
    public Changes getChanges(Solution before, Solution after) {
        final ChangesGenerationResult result = factories.stream()
                .map(factory -> generate(before, after, factory))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .min(Comparator.comparingInt(ChangesGenerationResult::size))
                .orElseThrow(RuntimeException::new);
        final List<CodeChange> changes = result.actions.stream()
                .map(action -> CodeChange.fromAction(action, result.mappings))
                .collect(Collectors.toList());
        return new Changes(before, after, changes);
    }

    private Optional<ChangesGenerationResult> generate(Solution before, Solution after,
                                                       BiFunction<ITree, ITree, Matcher> factory) {
        try {
            final ITree beforeTree = generator.buildTree(before);
            final ITree afterTree = generator.buildTree(after);
            final Matcher matcher = factory.apply(beforeTree, afterTree);
            matcher.match();
            final ActionGenerator generator = new ActionGenerator(beforeTree, afterTree, matcher.getMappings());
            return Optional.of(new ChangesGenerationResult(generator.generate(), matcher.getMappings()));
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    @Override
    public ASTGenerator getGenerator() {
        return generator;
    }

    private class ChangesGenerationResult {
        private final List<Action> actions;
        private final MappingStore mappings;

        private ChangesGenerationResult(List<Action> actions, MappingStore mappings) {
            this.actions = actions;
            this.mappings = mappings;
        }

        private int size() {
            return actions.size();
        }
    }
}

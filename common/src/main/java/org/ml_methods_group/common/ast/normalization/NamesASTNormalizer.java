package org.ml_methods_group.common.ast.normalization;

import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.common.ast.changes.MetadataKeys;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NamesASTNormalizer extends BasicASTNormalizer {
    public void normalize(TreeContext context, String code) {
        context.setRoot(new NamesProcessor(context, code).visit(context.getRoot()));
        context.validate();
    }

    private static class NamesProcessor extends StructureProcessor {
        private final ArrayDeque<Map<String, String>> aliases = new ArrayDeque<>();
        private final Map<String, Integer> totalCounters = new HashMap<>();
        private final ArrayDeque<Map<String, Integer>> counters = new ArrayDeque<>();
        private final String code;

        NamesProcessor(TreeContext context, String code) {
            super(context, code);
            this.code = code;
            aliases.add(new HashMap<>());
            counters.add(new HashMap<>());
        }

        void register(String name) {
            assert !typeDeclarations.isEmpty();
            register(name, typeDeclarations.peekLast());
        }

        void registerAsArray(String name, int arity) {
            assert !typeDeclarations.isEmpty();
            final String type = IntStream.range(0, arity)
                    .mapToObj(x -> "[]")
                    .collect(Collectors.joining("", typeDeclarations.peekLast(), ""));
            register(name, type);
        }

        void register(String name, String type) {
            super.register(name, type);
            final int id = totalCounters.compute(type, (t, count) -> (count == null ? 0 : count) + 1);
            counters.peekLast().compute(type, (t, count) -> (count == null ? 0 : count) + 1);
            final String alias = type + "@" + id;
            aliases.peekLast().put(name, alias);
        }

        private String getVariableAlias(String name) {
            final Iterator<Map<String, String>> iterator = aliases.descendingIterator();
            while (iterator.hasNext()) {
                final String alias = iterator.next().get(name);
                if (alias != null) {
                    return alias;
                }
            }
            return null;
        }

        void pushLayer() {
            super.pushLayer();
            aliases.addLast(new HashMap<>());
            counters.addLast(new HashMap<>());
        }

        void popLayer() {
            super.popLayer();
            aliases.pollLast();
            for (Map.Entry<String, Integer> entry : counters.pollLast().entrySet()) {
                totalCounters.compute(entry.getKey(), (t, count) -> count - entry.getValue());
            }
        }

        @Override
        protected ITree visitMyVariableName(ITree node) {
            final String oldLabel = node.getLabel();
            node.setLabel(getVariableAlias(oldLabel));
            node.setMetadata(MetadataKeys.ORIGINAL_NAME, oldLabel);
            return super.visitMyVariableName(node);
        }
    }
}

package org.ml_methods_group.core.changes;

import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.core.entities.NodeType;

import java.util.*;

import static org.ml_methods_group.core.changes.ASTUtils.getFirstChild;
import static org.ml_methods_group.core.entities.NodeType.*;

public class BasicASTNormalizer implements ASTNormalizer {

    @Override
    public void normalize(TreeContext context, String code) {
        context.setRoot(new Normalizer(context, code).visit(context.getRoot()));
        context.validate();
    }

    private static class Normalizer extends ASTProcessor {
        private final ArrayDeque<Map<String, String>> layers = new ArrayDeque<>();
        private final ArrayDeque<String> typeDeclarations = new ArrayDeque<>();
        private final String code;

        private Normalizer(TreeContext context, String code) {
            super(context);
            this.code = code;
            layers.add(new HashMap<>());
        }

        private void register(String name) {
            if (typeDeclarations.isEmpty()) {
                System.err.println(name);
                System.err.println(code);
            }
            assert !typeDeclarations.isEmpty();
            layers.peekLast().put(name, typeDeclarations.peekLast());
        }

        private void registerAsArray(String name) {
            assert !typeDeclarations.isEmpty();
            layers.peekLast().put(name, typeDeclarations.peekLast() + "[]");
        }

        private String getVariableType(String name) {
            final Iterator<Map<String, String>> iterator = layers.descendingIterator();
            while (iterator.hasNext()) {
                final String type = iterator.next().get(name);
                if (type != null) {
                    return type;
                }
            }
            return null;
        }

        private void pushLayer() {
            layers.addLast(new HashMap<>());
        }

        private void popLayer() {
            layers.pollLast();
        }

        private void pushTypeDeclaration(String type) {
            typeDeclarations.addLast(type);
        }

        private void popTypeDeclaration() {
            typeDeclarations.pollFirst();
        }

        @Override
        protected ITree visitFieldDeclaration(ITree node) {
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE, ARRAY_TYPE);
            assert type != null;
            pushTypeDeclaration(getTypeName(type));
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitLambdaExpression(ITree node) {
            pushLayer();
            pushTypeDeclaration("UnknownLambdaArgumentType");
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            popLayer();
            return result;
        }

        @Override
        protected ITree visitQualifiedName(ITree node) {
            return createNode(NodeType.MY_MEMBER_NAME,
                    node.getLabel().substring(node.getLabel().lastIndexOf('.') + 1));
        }

        @Override
        protected ITree visitVariableDeclarationFragment(ITree node) {
            final List<ITree> children = node.getChildren();
            assert 1 <= children.size() && children.size() <= 3;
            assert children.get(0).getType() == SIMPLE_NAME.ordinal();
            if (children.size() > 1 && children.get(1).getType() == NodeType.DIMENSION.ordinal()) {
                registerAsArray(children.get(0).getLabel());
            } else {
                register(children.get(0).getLabel());
            }
            final List<ITree> generated = new ArrayList<>();
            children.subList(1, children.size())
                    .forEach(tree -> generated.add(visit(tree)));
            node.setChildren(generated);
            return node;
        }

        @Override
        protected ITree visitVariableDeclarationStatement(ITree node) {
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE, ARRAY_TYPE);
            assert type != null;
            pushTypeDeclaration(getTypeName(type));
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitVariableDeclarationExpression(ITree node) {
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE, ARRAY_TYPE);
            assert type != null;
            pushTypeDeclaration(getTypeName(type));
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitSimpleType(ITree node) {
            node.setChildren(Collections.emptyList());
            return node;
        }

        @Override
        protected ITree visitSimpleName(ITree node) {
            //todo Maybe should use type info
            final String type = getVariableType(node.getLabel());
            if (type != null) {
                node.setLabel("@var");
            }
            return node;
        }

        @Override
        protected ITree visitMethodInvocation(ITree node) {
            final List<ITree> children = node.getChildren();
            final String text = code.substring(node.getPos(), node.getEndPos());
            final int bound;
            if (children.get(0).getType() != SIMPLE_NAME.ordinal()) {
                bound = 1;
            } else if (text.matches("(?s)[а-яА-Яa-zA-Z0-9_]+\\s*\\.\\s*(<[^>]+>)?\\s*[а-яА-Яa-zA-Z0-9_]+\\s*\\(.*")) {
                bound = 1;
            } else if (text.matches("(?s)(<[^>]+>)?\\s*[а-яА-Яa-zA-Z0-9_]+\\s*\\(.*")) {
                bound = 0;
            } else {
                System.err.println(code);
                throw new RuntimeException("Unexpected situation: " + text);
            }
            final List<ITree> generated = new ArrayList<>(children.size());
            boolean flag = false;
            for (int i = 0; i < children.size(); i++) {
                final ITree child = children.get(i);
                if (!flag && i >= bound && child.getType() == NodeType.SIMPLE_NAME.ordinal()) {
                    flag = true;
                    generated.add(createNode(NodeType.MY_MEMBER_NAME, child.getLabel()));
                    continue;
                }
                final ITree result = visit(child);
                if (result != null) {
                    generated.add(child);
                }
            }
            node.setChildren(generated);
            return node;
        }

        @Override
        protected ITree visitJavadoc(ITree node) {
            return null;
        }

        @Override
        protected ITree visitLineComment(ITree node) {
            return null;
        }

        @Override
        protected ITree visitBlockComment(ITree node) {
            return null;
        }

        @Override
        protected ITree visitSingleVariableDeclaration(ITree node) {
            final ITree name = getFirstChild(node, SIMPLE_NAME);
            final ITree type = getFirstChild(node, SIMPLE_TYPE, PARAMETERIZED_TYPE, PRIMITIVE_TYPE,
                    ARRAY_TYPE, UNION_TYPE);
            if (type == null) {
                System.err.println(code);
                System.err.println(code.substring(node.getPos(), node.getEndPos()));
            }
            assert name != null;
            assert type != null;
            pushTypeDeclaration(getTypeName(type));
            register(name.getLabel());
            final ITree result = defaultVisit(node);
            popTypeDeclaration();
            return result;
        }

        @Override
        protected ITree visitBlock(ITree node) {
            pushLayer();
            final ITree result = defaultVisit(node);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitForStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 3);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitWhileStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 1);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitIfStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 1, 2);
            popLayer();
            return result;
        }

        @Override
        protected ITree visitEnhancedForStatement(ITree node) {
            pushLayer();
            final ITree result = checkBlocks(node, 2);
            popLayer();
            return result;
        }

        private static String getTypeName(ITree node) {
            final NodeType type = NodeType.valueOf(node.getType());
            if (type == null) {
                throw new RuntimeException();
            }
            switch (type) {
                case PRIMITIVE_TYPE:
                    return node.getLabel();
                case SIMPLE_TYPE:
                    final String label = node.getLabel();
                    return label.substring(label.lastIndexOf('.') + 1);
                case PARAMETERIZED_TYPE:
                    final ITree child = getFirstChild(node, SIMPLE_TYPE);
                    assert child != null;
                    return child.getLabel();
                case ARRAY_TYPE:
                    return node.getLabel();
                case UNION_TYPE:
                    return "UnionThrowable";
            }
            throw new RuntimeException("Unexpected type!");
        }

        private ITree checkBlocks(ITree node, int... positions) {
            final BitSet targets = new BitSet();
            Arrays.stream(positions).forEach(targets::set);
            final List<ITree> children = node.getChildren();
            final List<ITree> generated = new ArrayList<>();
            for (int i = 0; i < children.size(); i++) {
                final ITree child = visit(children.get(i));
                if (targets.get(i) && child.getType() != NodeType.BLOCK.ordinal()) {
                    final ITree wrapper = createNode(NodeType.BLOCK, "");
                    wrapper.addChild(child);
                    generated.add(wrapper);
                } else {
                    generated.add(child);
                }

            }
            node.setChildren(generated);
            return node;
        }
    }
}

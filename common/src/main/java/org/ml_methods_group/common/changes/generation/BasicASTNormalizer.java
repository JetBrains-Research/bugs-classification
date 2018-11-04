package org.ml_methods_group.common.changes.generation;

import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.common.changes.NodeType;

import java.util.*;
import java.util.stream.Collectors;


public class BasicASTNormalizer implements ASTNormalizer {

    public void normalize(TreeContext context, String code) {
        context.setRoot(new Normalizer(context, code).visit(context.getRoot()));
        context.validate();
    }

    private static class Normalizer extends ASTProcessor {
        private final String code;

        private Normalizer(TreeContext context, String code) {
            super(context);
            this.code = code;
        }

        @Override
        protected ITree visitQualifiedName(ITree node) {
            assert node.getChildren().isEmpty();
            if (node.getParent().getType() == NodeType.PACKAGE_DECLARATION.ordinal()) {
                return createNode(NodeType.SIMPLE_NAME, node.getLabel());
            }
            final String label = node.getLabel();
            final ITree path = createNode(NodeType.SIMPLE_NAME, label.substring(0, label.lastIndexOf('.')));
            final ITree member = createNode(NodeType.MY_MEMBER_NAME, label.substring(label.lastIndexOf('.') + 1));
            member.setChildren(Collections.singletonList(path));

            return visit(member);
        }

        @Override
        protected ITree visitVariableDeclarationFragment(ITree node) {
            final List<ITree> children = node.getChildren();
            assert children.get(0).getType() == NodeType.SIMPLE_NAME.ordinal();
            final List<ITree> generated = new ArrayList<>();
            children.subList(1, children.size())
                    .forEach(tree -> generated.add(visit(tree)));
            node.setChildren(generated);
            return node;
        }

        @Override
        protected ITree visitSimpleType(ITree node) {
            final String label = node.getLabel();
            node.setLabel(label.substring(label.lastIndexOf('.') + 1));
            node.setChildren(Collections.emptyList());
            return node;
        }

        @Override
        protected ITree visitExpressionStatement(ITree node) {
            assert node.getChildren().size() == 1;
            return visit(node.getChild(0));
        }

        @Override
        protected ITree visitMethodInvocation(ITree node) {
            final List<ITree> children = node.getChildren();
            final String text = code.substring(node.getPos(), node.getEndPos());
            final int bound;
            if (children.get(0).getType() != NodeType.SIMPLE_NAME.ordinal()) {
                bound = 0;
            } else if (text.matches("(?s)\\s*[a-zA-Z0-9_]+\\s*\\(.*")) {
                bound = 0;
            } else {
                bound = 1;
            }
            final List<ITree> generated = new ArrayList<>(children.size());
            final List<ITree> arguments = new ArrayList<>(children.size());
            boolean flag = false;
            for (int i = 0; i < children.size(); i++) {
                final ITree child = children.get(i);
                if (!flag && i >= bound && child.getType() == NodeType.SIMPLE_NAME.ordinal()) {
                    flag = true;
                    node.setLabel(child.getLabel());
                    continue;
                }
                final ITree result = visit(child);
                if (result == null) {
                    continue;
                }
                if (flag) {
                    arguments.add(result);
                } else {
                    generated.add(result);
                }
            }
            final ITree argumentsNode = createNode(NodeType.MY_METHOD_INVOCATION_ARGUMENTS, node.getLabel());
            argumentsNode.setChildren(arguments);
            generated.add(argumentsNode);
            node.setChildren(generated);
            return node;
        }

        @Override
        protected ITree visitImportDeclaration(ITree node) {
            final List<ITree> children = node.getChildren();
            assert children.size() == 1;
            final List<ITree> generated = new ArrayList<>(2);
            final String name = children.get(0).getLabel();
            final int separatorIndex = name.lastIndexOf('.');
            if (separatorIndex == -1) {
                generated.add(visit(createNode(NodeType.SIMPLE_TYPE, name)));
            } else {
                final String path = name.substring(0, separatorIndex);
                final String type = name.substring(separatorIndex + 1);
                generated.add(visit(createNode(NodeType.MY_PATH_NAME, path)));
                if (type.equals("*")) {
                    generated.add(visit(createNode(NodeType.MY_ALL_CLASSES, "")));
                } else {
                    generated.add(visit(createNode(NodeType.SIMPLE_TYPE, type)));
                }
            }
            generated.removeIf(Objects::isNull);
            node.setChildren(generated);
            return super.visitImportDeclaration(node);
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
        protected ITree visitForStatement(ITree node) {
            return checkBlocks(node, 3);
        }

        @Override
        protected ITree visitWhileStatement(ITree node) {
            return checkBlocks(node, 1);
        }

        @Override
        protected ITree visitIfStatement(ITree node) {
            return checkBlocks(node, 1, 2);
        }

        @Override
        protected ITree visitEnhancedForStatement(ITree node) {
            return checkBlocks(node, 2);
        }

        @Override
        protected ITree visitUnionType(ITree node) {
            final List<ITree> children = node.getChildren()
                    .stream()
                    .map(this::visit)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
            node.setChildren(children);
            final String label = Arrays.stream(node.getLabel().split("\\|"))
                    .sorted()
                    .collect(Collectors.joining("|"));
            node.setLabel(label);
            return node;
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

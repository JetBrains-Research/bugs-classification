package org.ml_methods_group.core.vectorization;


import org.ml_methods_group.core.entities.ChangeType;
import org.ml_methods_group.core.entities.CodeChange;

import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.entities.ChangeType.*;
import static org.ml_methods_group.core.vectorization.BasicEncodingStrategy.ChangeAttribute.*;

public class BasicEncodingStrategy implements EncodingStrategy {

    private final Map<String, Integer> dictionary = new HashMap<>();
    private final Set<ChangeAttribute> attributes;
    private final Set<ChangeType> types;
    private final int encodingType;

    public BasicEncodingStrategy(int encodingType, List<ChangeAttribute> attributes, List<ChangeType> types) {
        this.encodingType = encodingType;
        this.types = new HashSet<>(types);
        this.attributes = new HashSet<>(attributes);
        this.attributes.add(ENCODING_TYPE);
    }

    public BasicEncodingStrategy(List<ChangeAttribute> attributes, List<ChangeType> types) {
        this(1, attributes, types);
    }


    @Override
    public long encode(CodeChange change) {
        if (!types.contains(change.getChangeType())) {
            return 0;
        }
        long result = 0;
        for (ChangeAttribute attribute : attributes) {
            final int attributeCode = getAttributeCode(change, attribute);
            final long attributeShiftedCode = encode(attributeCode, attribute.offset, attribute.limit);
            if ((result & attributeShiftedCode) != 0) {
                throw new RuntimeException("Unreachable situation");
            }
            result |= attributeShiftedCode;
        }
        return result;
    }

    private static long encode(int code, int shift, int limit) {
        if (code >= (1 << limit)) {
            throw new RuntimeException("Encoding overflow!");
        }
        return (long) code << shift;
    }

    private static int decode(long code, int shift, int limit) {
        return (int) ((code >>> shift) & ((1 << limit) - 1));
    }

    private int getAttributeCode(CodeChange change, ChangeAttribute attribute) {
        if (!attribute.isApplicable(change.getChangeType())) {
            return 0;
        }
        switch (attribute) {
            case CHANGE_TYPE:
                return change.getChangeType().ordinal();
            case NODE_TYPE:
                return 1 + change.getNodeType().ordinal();
            case PARENT_TYPE:
                return 1 + change.getParentType().ordinal();
            case PARENT_OF_PARENT_TYPE:
                return 1 + change.getParentOfParentType().ordinal();
            case OLD_PARENT_TYPE:
                return 1 + change.getOldParentType().ordinal();
            case OLD_PARENT_OF_PARENT_TYPE:
                return 1 + change.getOldParentOfParentType().ordinal();
            case LABEL_TYPE:
                return 1 + dictionary.computeIfAbsent(change.getLabel(), label -> dictionary.size());
            case OLD_LABEL_TYPE:
                return 1 + dictionary.getOrDefault(change.getOldLabel(), 0);
            case ENCODING_TYPE:
                return 1 + encodingType;
        }
        throw new RuntimeException("Unexpected attribute type");
    }

    public enum ChangeAttribute {
        CHANGE_TYPE(2, 0, ChangeType.values()),
        ENCODING_TYPE(7, 2, ChangeType.values()),
        NODE_TYPE(7, 9, ChangeType.values()),
        PARENT_TYPE(7, 16, ChangeType.values()),
        PARENT_OF_PARENT_TYPE(7, 23, ChangeType.values()),
        LABEL_TYPE(16, 30, ChangeType.values()),
        OLD_PARENT_TYPE(7, 46, MOVE),
        OLD_PARENT_OF_PARENT_TYPE(7, 53, MOVE),
        OLD_LABEL_TYPE(16, 46, UPDATE);

        final int offset;
        final int limit;
        final BitSet mask;

        ChangeAttribute(int limit, int offset, ChangeType... types) {
            this.offset = offset;
            this.limit = limit;
            this.mask = new BitSet();
            Arrays.stream(types)
                    .mapToInt(ChangeType::ordinal)
                    .forEach(mask::set);
        }

        boolean isApplicable(ChangeType type) {
            return isApplicable(type.ordinal());
        }

        boolean isApplicable(int type) {
            return mask.get(type);
        }
    }

    public static List<EncodingStrategy> defaultStrategies() {
        return Arrays.asList(
                new BasicEncodingStrategy(2,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE),
                        Arrays.asList(DELETE, INSERT)),
                new BasicEncodingStrategy(3,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, LABEL_TYPE),
                        Arrays.asList(DELETE, INSERT)),
                new BasicEncodingStrategy(4,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, OLD_PARENT_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(5,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(6,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE,
                                OLD_PARENT_OF_PARENT_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(7,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, OLD_PARENT_TYPE,
                                LABEL_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(8,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE,
                                OLD_PARENT_TYPE, OLD_PARENT_OF_PARENT_TYPE, LABEL_TYPE),
                        Collections.singletonList(MOVE)),
                new BasicEncodingStrategy(9,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, PARENT_OF_PARENT_TYPE, LABEL_TYPE),
                        Collections.singletonList(UPDATE)),
                new BasicEncodingStrategy(10,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE, LABEL_TYPE, OLD_LABEL_TYPE),
                        Collections.singletonList(UPDATE)),
                new BasicEncodingStrategy(11,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, LABEL_TYPE, OLD_LABEL_TYPE),
                        Collections.singletonList(UPDATE)),
                new BasicEncodingStrategy(12,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE),
                        Arrays.asList(ChangeType.values())),
                new BasicEncodingStrategy(13,
                        Arrays.asList(CHANGE_TYPE, NODE_TYPE, PARENT_TYPE),
                        Arrays.asList(ChangeType.values())),
                new BasicEncodingStrategy(14,
                        Arrays.asList(CHANGE_TYPE, LABEL_TYPE),
                        Arrays.asList(ChangeType.values()))
        );
    }
}

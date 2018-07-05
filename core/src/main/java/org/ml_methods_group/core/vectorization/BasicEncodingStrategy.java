package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.changes.ChangeType;
import org.ml_methods_group.core.changes.NodeType;

import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.vectorization.BasicEncodingStrategy.ChangeAttribute.*;

public class BasicEncodingStrategy implements EncodingStrategy {

    private final Map<String, Integer> dictionary;
    private final Map<Integer, String> reversed;
    private final Set<ChangeAttribute> attributes;
    private final Set<ChangeType> types;
    private final int encodingType;

    public BasicEncodingStrategy(Map<String, Integer> dictionary, int encodingType,
                                 List<ChangeAttribute> attributes, List<ChangeType> types) {
        this.dictionary = dictionary;
        this.encodingType = encodingType;
        this.types = new HashSet<>(types);
        this.attributes = new HashSet<>(attributes);
        this.attributes.add(ENCODING_TYPE);
        this.reversed = dictionary.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey, (a, b) -> a + "|" + b));
    }

    public BasicEncodingStrategy(Map<String, Integer> dictionary,
                                 List<ChangeAttribute> attributes, List<ChangeType> types) {
        this(dictionary, 1, attributes, types);
    }


    @Override
    public long encode(AtomicChange change) {
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

    @Override
    public ChangeCodeWrapper decode(long code) {
        if (encodingType != decode(code, ENCODING_TYPE)) {
            throw new IllegalArgumentException();
        }
        return new ChangeCodeWrapper(code, encodingType,
                ChangeType.valueOf(decode(code, CHANGE_TYPE)),
                NodeType.valueOf(decode(code, NODE_TYPE)),
                NodeType.valueOf(decode(code, PARENT_TYPE)),
                NodeType.valueOf(decode(code, PARENT_OF_PARENT_TYPE)),
                NodeType.valueOf(decode(code, OLD_PARENT_TYPE)),
                NodeType.valueOf(decode(code, OLD_PARENT_OF_PARENT_TYPE)),
                reversed.getOrDefault(decode(code, LABEL_TYPE), "-"),
                reversed.getOrDefault(decode(code, OLD_LABEL_TYPE), "-"));
    }

    private static long encode(int code, int shift, int limit) {
        if (code >= (1 << limit)) {
            throw new RuntimeException("Encoding overflow!");
        }
        return (long) code << shift;
    }

    private static int decode(long code, ChangeAttribute attribute) {
        final int action = decode(code, ChangeAttribute.CHANGE_TYPE.offset, ChangeAttribute.CHANGE_TYPE.limit);
        if (attribute == CHANGE_TYPE) {
            return action;
        } else if (!attribute.isApplicable(action)) {
            return -1;
        }
        return decode(code, attribute.offset, attribute.limit) - 1;
    }

    private static int decode(long code, int shift, int limit) {
        return (int) ((code >>> shift) & ((1 << limit) - 1));
    }

    private int getAttributeCode(AtomicChange change, ChangeAttribute attribute) {
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
                return 1 + dictionary.getOrDefault(change.getLabel(), 0);
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
        OLD_PARENT_TYPE(7, 46, ChangeType.MOVE),
        OLD_PARENT_OF_PARENT_TYPE(7, 53, ChangeType.MOVE),
        OLD_LABEL_TYPE(16, 46, ChangeType.UPDATE);

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
}

package ru.spbau.mit.lobanov.changes;

import java.util.*;

import static ru.spbau.mit.lobanov.changes.EncodingStrategy.ChangeAttribute.CHANGE_TYPE;

public class EncodingStrategy {
    private final Map<String, Integer> labelCodes;
    private final Set<ChangeAttribute> attributes;

    public EncodingStrategy(Map<String, Integer> labelCodes, ChangeAttribute... attributes) {
        this(labelCodes, Arrays.asList(attributes));
    }

    public EncodingStrategy(Map<String, Integer> labelCodes, Collection<ChangeAttribute> attributes) {
        this.labelCodes = labelCodes;
        this.attributes = new HashSet<>(attributes);
    }

    public long encode(AtomicChange change) {
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

    public int decodeAttribute(long code, ChangeAttribute attribute) {
        if (!attributes.contains(attribute)) {
            return -1;
        }
        final int type = decode(code, CHANGE_TYPE.offset, CHANGE_TYPE.limit) - 1;
        return attribute.isApplicable(type) ? decode(code, attribute.offset, attribute.limit) - 1: 0;
    }

    private int getLabelCode(String label) {
        return labelCodes.getOrDefault(label, 0);
    }

    private int getAttributeCode(AtomicChange change, ChangeAttribute attribute) {
        if (!attribute.isApplicable(change.getChangeType())) {
            return 0;
        }
        switch (attribute) {
            case CHANGE_TYPE:
                return 1 + change.getChangeType().ordinal();
            case NODE_TYPE:
                return 1 + change.getNodeType();
            case PARENT_TYPE:
                return 1 + change.getParentType();
            case PARENT_OF_PARENT_TYPE:
                return 1 + change.getParentOfParentType();
            case OLD_PARENT_TYPE:
                return 1 + change.getOldParentType();
            case OLD_PARENT_OF_PARENT_TYPE:
                return 1 + change.getOldParentOfParentType();
            case LABEL_TYPE:
                return 1 + getLabelCode(change.getLabel());
            case OLD_LABEL_TYPE:
                return 1 + getLabelCode(change.getOldLabel());
        }
        throw new RuntimeException("Unexpected attribute type");
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

    public enum ChangeAttribute {
        CHANGE_TYPE(3, 0, ChangeType.values()),
        NODE_TYPE(7, 3, ChangeType.values()),
        PARENT_TYPE(7, 10, ChangeType.values()),
        PARENT_OF_PARENT_TYPE(7, 17, ChangeType.values()),
        LABEL_TYPE(16, 24, ChangeType.values()),
        OLD_PARENT_TYPE(7, 40, ChangeType.MOVE),
        OLD_PARENT_OF_PARENT_TYPE(7, 47, ChangeType.MOVE),
        OLD_LABEL_TYPE(16, 40, ChangeType.UPDATE);

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

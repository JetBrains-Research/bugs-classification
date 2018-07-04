package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.changes.AtomicChange.ChangeType;
import org.ml_methods_group.core.vectorization.EncodingStrategy;

import java.util.*;

public class ChangeEncodingStrategy implements EncodingStrategy<AtomicChange> {

    private final Map<String, Long> dictionary;
    private final Set<ChangeAttribute> attributes;
    private final int encodingType;

    public ChangeEncodingStrategy(Map<String, Long> dictionary, int encodingType, ChangeAttribute... attributes) {
        this.dictionary = dictionary;
        this.encodingType = encodingType;
        this.attributes = new HashSet<>(Arrays.asList(attributes));
    }

    public ChangeEncodingStrategy(Map<String, Long> dictionary, ChangeAttribute... attributes) {
        this(dictionary, 1, attributes);
    }

    @Override
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

    private static long encode(int code, int shift, int limit) {
        if (code >= (1 << limit)) {
            throw new RuntimeException("Encoding overflow!");
        }
        return (long) code << shift;
    }

    public static int decode(long code, ChangeAttribute attribute) {
        final int action = decode(code, ChangeAttribute.CHANGE_TYPE.offset, ChangeAttribute.CHANGE_TYPE.limit);
        if (!attribute.isApplicable(action)) {
            return -1;
        }
        return decode(code, attribute.offset, attribute.limit);
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
                return 1 + dictionary.getOrDefault(change.getLabel(), 0L).intValue();
            case OLD_LABEL_TYPE:
                return 1 + dictionary.getOrDefault(change.getOldLabel(), 0L).intValue();
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

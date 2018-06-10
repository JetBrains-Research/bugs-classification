package ru.spbau.mit.lobanov.changes;

public abstract class AtomicChange {

    static final int CHANGE_TYPE_OFFSET = 0;
    static final int NODE_TYPE_OFFSET = 1;
    static final int PARENT_TYPE_OFFSET = 2;
    static final int PARENT_OF_PARENT_TYPE_OFFSET = 3;
    static final int LABEL_OFFSET = 4;
    static final int OLD_PARENT_TYPE_OFFSET = 5;
    static final int OLD_PARENT_OF_PARENT_TYPE_OFFSET = 6;
    static final int OLD_LABEL_OFFSET = 7;

    private final int nodeType;
    private final int parentType;
    private final int parentOfParentType;
    private final String label;

    public AtomicChange(int nodeType, int parentType, int parentOfParentType, String label) {
        this.nodeType = nodeType;
        this.parentType = parentType;
        this.parentOfParentType = parentOfParentType;
        this.label = ChangeUtils.normalize(label);
    }

    public void storeData(Object[] dst, int offset) {
        dst[offset + CHANGE_TYPE_OFFSET] = getChangeType().ordinal();
        dst[offset + NODE_TYPE_OFFSET] = nodeType;
        dst[offset + PARENT_TYPE_OFFSET] = parentType;
        dst[offset + PARENT_OF_PARENT_TYPE_OFFSET] = parentOfParentType;
        dst[offset + LABEL_OFFSET] = label;
        dst[offset + OLD_PARENT_TYPE_OFFSET] = getOldParentType();
        dst[offset + OLD_PARENT_OF_PARENT_TYPE_OFFSET] = getOldParentOfParentType();
        dst[offset + OLD_LABEL_OFFSET] = getOldLabel();
    }

    public abstract ChangeType getChangeType();

    public int getNodeType() {
        return nodeType;
    }

    public int getParentType() {
        return parentType;
    }

    public int getParentOfParentType() {
        return parentOfParentType;
    }

    public String getLabel() {
        return label;
    }

    public int getOldParentType() {
        return 0;
    }

    public int getOldParentOfParentType() {
        return 0;
    }

    public String getOldLabel() {
        return "";
    }
}

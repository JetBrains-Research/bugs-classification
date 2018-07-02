package org.ml_methods_group.core.changes;

public interface AtomicChange {

    ChangeType getChangeType();

    default int getNodeType() {
        return -1;
    }

    default int getParentType() {
        return -1;
    }

    default int getParentOfParentType() {
        return -1;
    }

    default String getLabel() {
        return "";
    }

    default int getOldParentType() {
        return -1;
    }

    default int getOldParentOfParentType() {
        return -1;
    }

    default String getOldLabel() {
        return "";
    }

    enum ChangeType {
        DELETE, INSERT, MOVE, UPDATE
    }
}

package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.preparation.LabelType;

public class LabelWrapper {
    private final String label;
    private final LabelType type;
    private final int id;

    public LabelWrapper(String label, LabelType type, int id) {
        this.label = label;
        this.type = type;
        this.id = id;
    }

    public String getLabel() {
        return label;
    }

    public int getId() {
        return id;
    }

    public LabelType getType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LabelWrapper that = (LabelWrapper) o;

        return id == that.id && label.equals(that.label) && type == that.type;
    }

    @Override
    public int hashCode() {
        int result = label.hashCode();
        result = 31 * result + type.hashCode();
        result = 31 * result + id;
        return result;
    }
}

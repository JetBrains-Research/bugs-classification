package org.ml_methods_group.core.vectorization;

public class LabelWrapper {
    private final String label;
    private final int id;

    public LabelWrapper(String label, int id) {
        this.label = label;
        this.id = id;
    }

    public String getLabel() {
        return label;
    }

    public int getId() {
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LabelWrapper that = (LabelWrapper) o;

        return id == that.id && label.equals(that.label);
    }

    @Override
    public int hashCode() {
        int result = label.hashCode();
        result = 31 * result + id;
        return result;
    }
}

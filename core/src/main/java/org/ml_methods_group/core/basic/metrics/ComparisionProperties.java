package org.ml_methods_group.core.basic.metrics;

import org.ml_methods_group.core.changes.NodeType;

import java.util.Arrays;

public class ComparisionProperties {
    private final double[] labelImportance;
    private final double[] oldLabelImportance;
    private final double[] childrenImportance;
    private final double[] brothersImportance;
    private final double[] asParentImportance;
    private final double[] asParentOfParentImportance;

    public ComparisionProperties(double defaultLabelImportance, double defaultOldLabelImportance,
                                 double defaultAsParentImportance, double defaultAsParentOfParentImportance,
                                 double defaultChildrenImportance, double defaultBrothersImportance) {
        final int size = NodeType.values().length;
        labelImportance = new double[size];
        Arrays.fill(labelImportance, defaultLabelImportance);
        oldLabelImportance = new double[size];
        Arrays.fill(oldLabelImportance, defaultOldLabelImportance);
        asParentImportance = new double[size];
        Arrays.fill(asParentImportance, defaultAsParentImportance);
        asParentOfParentImportance = new double[size];
        Arrays.fill(asParentOfParentImportance, defaultAsParentOfParentImportance);
        childrenImportance = new double[size];
        Arrays.fill(childrenImportance, defaultChildrenImportance);
        brothersImportance = new double[size];
        Arrays.fill(brothersImportance, defaultBrothersImportance);
    }

    public void setLabelImportance(NodeType type, double labelImportance, double oldLabelImportance) {
        this.labelImportance[type.ordinal()] = labelImportance;
        this.oldLabelImportance[type.ordinal()] = oldLabelImportance;
    }

    public void setAsParentImportance(NodeType type, double asParentImportance) {
        this.asParentImportance[type.ordinal()] = asParentImportance;
    }

    public void setAsParentOfParentImportance(NodeType type, double asParentOfParentImportance) {
        this.asParentOfParentImportance[type.ordinal()] = asParentOfParentImportance;
    }

    public void setChildrenImportance(NodeType type, double childrenImportance) {
        this.childrenImportance[type.ordinal()] = childrenImportance;
    }

    public void setBrothersImportance(NodeType type, double brothersImportance) {
        this.brothersImportance[type.ordinal()] = brothersImportance;
    }

    public double getLabelImportance(NodeType type) {
        return labelImportance[type.ordinal()];
    }

    public double getOldLabelImportance(NodeType type) {
        return oldLabelImportance[type.ordinal()];
    }

    public double getAsParentImportance(NodeType type) {
        return asParentImportance[type.ordinal()];
    }

    public double getAsParentOfParentImportance(NodeType type) {
        return asParentOfParentImportance[type.ordinal()];
    }

    public double getChildrenImportance(NodeType type) {
        return childrenImportance[type.ordinal()];
    }

    public double getBrothersImportance(NodeType type) {
        return brothersImportance[type.ordinal()];
    }
}

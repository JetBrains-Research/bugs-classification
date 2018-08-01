package org.ml_methods_group.core.entities;

import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

@DataClass(defaultStorageName = "decisions_cache")
public class CachedDecision {
    @DataField
    private final int valueId;
    @DataField
    private final int targetId;

    public CachedDecision() {
        this(0, 0);
    }

    public CachedDecision(int valueId, int targetId) {
        this.valueId = valueId;
        this.targetId = targetId;
    }

    public int getValueId() {
        return valueId;
    }

    public int getTargetId() {
        return targetId;
    }
}

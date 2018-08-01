package org.ml_methods_group.core.entities;

import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

@DataClass(defaultStorageName = "distance_cache")
public class CachedDistance {
    @DataField
    private final int firstId;
    @DataField
    private final int secondId;
    @DataField
    private final double distance;

    public CachedDistance() {
        this(0, 0, 0);
    }

    public CachedDistance(int firstId, int secondId, double distance) {
        this.firstId = firstId;
        this.secondId = secondId;
        this.distance = distance;
    }

    public int getFirstId() {
        return firstId;
    }

    public int getSecondId() {
        return secondId;
    }

    public double getDistance() {
        return distance;
    }
}

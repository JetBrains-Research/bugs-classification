package org.ml_methods_group.testing.markers;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.markers.Marker;

import java.util.Collections;
import java.util.List;

public class CachedExtrapolationMarker<V, M> extends ExtrapolationMarker<V, M> {
    private final CacheMarker<V, M> marker;
    private final Marker<Cluster<V>, M> oracle;
    private final int bound;

    public CachedExtrapolationMarker(CacheMarker<V, M> marker, int bound, Marker<Cluster<V>, M> oracle) {
        super(marker, bound);
        this.bound = bound;
        this.marker = marker;
        this.oracle = oracle;
    }

    @Override
    protected M onFail(Cluster<V> cluster) {
        final M mark = oracle.mark(cluster);
        if (mark != null) {
            final List<V> values = cluster.elementsCopy();
            Collections.shuffle(values);
            values.subList(0, Math.min(values.size(), bound))
                    .forEach(value -> marker.cacheMark(value, mark));
        }
        return mark;
    }
}

package org.ml_methods_group.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Scenario<V, M> {
    private final Clusterer<V> clusterer;
    private final Classifier<V, M> classifier;
    private final Marker<Cluster<V>, M> marker;
    private final List<V> samples;

    public Scenario(Clusterer<V> clusterer, Classifier<V, M> classifier,
                    Marker<Cluster<V>, M> marker, List<V> samples) {
        this.clusterer = clusterer;
        this.classifier = classifier;
        this.marker = marker;
        this.samples = samples;
    }

    public void run() {
        final List<Cluster<V>> clusters = clusterer.buildClusters(samples);
        final Map<Cluster<V>, M> marks = new HashMap<>();
        for (Cluster<V> cluster : clusters) {
            final M mark = marker.mark(cluster);
            if (mark != null) {
                marks.put(cluster, mark);
            }
        }
        classifier.train(marks);
    }

    public Classifier<V, M> getClassifier() {
        return classifier;
    }
}

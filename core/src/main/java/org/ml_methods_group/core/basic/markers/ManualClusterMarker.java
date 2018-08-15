package org.ml_methods_group.core.basic.markers;

import org.ml_methods_group.core.entities.Solution;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ManualClusterMarker extends AbstractManualMarker<List<Solution>, String> {

    private final int elementsBound;

    public ManualClusterMarker(InputStream input, OutputStream output, int elementsBound) {
        super(input, output);
        this.elementsBound = elementsBound;
    }

    public ManualClusterMarker(int elementsBound) {
        super(System.in, System.out);
        this.elementsBound = elementsBound;
    }

    @Override
    protected String valueToString(List<Solution> value) {
        final List<Solution> buffer = new ArrayList<>(value);
        Collections.shuffle(buffer);
        final StringBuilder builder = new StringBuilder();
        builder.append("Cluster: (Size: ").append(buffer.size()).append(")").append(System.lineSeparator());
        buffer.stream()
                .limit(elementsBound)
                .forEach(solution -> {
                    builder.append("Session id: ").append(solution.getSessionId()).append(System.lineSeparator());
                    builder.append(solution.getCode()).append(System.lineSeparator()).append(System.lineSeparator());
                });
        return builder.toString();
    }

    @Override
    protected String stringToMark(String token) {
        return token.toLowerCase();
    }
}

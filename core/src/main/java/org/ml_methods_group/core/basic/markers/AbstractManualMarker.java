package org.ml_methods_group.core.basic.markers;

import org.ml_methods_group.core.Marker;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Scanner;

public abstract class AbstractManualMarker<V, M> implements Marker<V, M> {
    private final Scanner input;
    private final PrintStream output;

    protected AbstractManualMarker(InputStream input, OutputStream output) {
        this.input = new Scanner(input);
        this.output = new PrintStream(output);
    }

    @Override
    public M mark(V value) {
        output.println("Mark requested for element:");
        output.println(valueToString(value));
        output.println("Your mark:");
        output.flush();
        final String token = input.next();
        return token.equals("-") ? null : stringToMark(token);
    }

    protected abstract String valueToString(V value);
    protected abstract M stringToMark(String token);
}

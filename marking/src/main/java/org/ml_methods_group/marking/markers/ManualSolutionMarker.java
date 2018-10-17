package org.ml_methods_group.marking.markers;

import org.ml_methods_group.common.Solution;

import java.io.InputStream;
import java.io.OutputStream;

public class ManualSolutionMarker extends AbstractManualMarker<Solution, String> {

    public ManualSolutionMarker(InputStream input, OutputStream output) {
        super(input, output);
    }

    public ManualSolutionMarker() {
        super(System.in, System.out);
    }

    @Override
    protected String valueToString(Solution value) {
        return "Session id:  " + value.getSessionId() + System.lineSeparator() +
                "Solution id: " + value.getSolutionId() + System.lineSeparator() +
                "Code:" + System.lineSeparator() +
                value.getCode() + System.lineSeparator();
    }

    @Override
    protected String stringToMark(String token) {
        return token.toLowerCase();
    }
}

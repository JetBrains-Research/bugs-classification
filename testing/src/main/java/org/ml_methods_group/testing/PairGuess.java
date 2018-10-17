package org.ml_methods_group.testing;

public enum PairGuess {
    SIMILAR, NEUTRAL, DIFFERENT;

    private static final PairGuess[] buffer = values();

    public static PairGuess valueOf(int value) {
        return value == -1 ? null : buffer[value];
    }
}

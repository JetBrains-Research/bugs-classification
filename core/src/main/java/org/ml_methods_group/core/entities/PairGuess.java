package org.ml_methods_group.core.entities;

public enum PairGuess {
    VERY_SIMILAR, SIMILAR, NEUTRAL, DIFFERENT, VERY_DIFFERENT;

    private static final PairGuess[] buffer = values();

    public static PairGuess valueOf(int value) {
        return value == -1 ? null : buffer[value];
    }
}

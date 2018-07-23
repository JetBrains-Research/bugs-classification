package org.ml_methods_group.core.entities;

import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

@DataClass(defaultStorageName = "tests")
public class TestPair {
    @DataField
    private final int firstSessionId;
    @DataField
    private final int secondSessionId;
    @DataField
    private final PairGuess guess;

    public TestPair() {
        this(0, 0, null);
    }

    public TestPair(int firstSessionId, int secondSessionId, PairGuess guess) {
        this.firstSessionId = firstSessionId;
        this.secondSessionId = secondSessionId;
        this.guess = guess;
    }

    public int getFirstSessionId() {
        return firstSessionId;
    }

    public int getSecondSessionId() {
        return secondSessionId;
    }

    public PairGuess getGuess() {
        return guess;
    }
}

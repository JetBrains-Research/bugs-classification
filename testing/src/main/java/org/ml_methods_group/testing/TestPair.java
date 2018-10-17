package org.ml_methods_group.testing;


public class TestPair {
    private final int firstSessionId;
    private final int secondSessionId;
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

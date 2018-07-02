package org.ml_methods_group.core;

public interface Solution {
    String getCode();

    int getProblemId();

    int getSessionId();

    Verdict getVerdict();

    enum Verdict {OK, FAIL, UNKNOWN}
}

package org.ml_methods_group.database.proxy;

import org.ml_methods_group.core.Solution;
import org.ml_methods_group.database.primitives.Table;

import static org.ml_methods_group.core.Solution.Verdict.FAIL;

public class ProxySolution implements Solution {

    private final Table codes;
    private final int sessionId;
    private final Verdict verdict;

    public ProxySolution(Table codes, int sessionId, Verdict verdict) {
        this.codes = codes;
        this.sessionId = sessionId;
        this.verdict = verdict;
    }

    @Override
    public String getCode() {
        try {
            return codes.findFirst(sessionId + (verdict == FAIL ? "_0" : "_1")).getStringValue("code");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int getProblemId() {
        try {
            return codes.findFirst(sessionId + (verdict == FAIL ? "_0" : "_1")).getIntValue("problem");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int getSessionId() {
        return sessionId;
    }

    @Override
    public Verdict getVerdict() {
        return verdict;
    }
}

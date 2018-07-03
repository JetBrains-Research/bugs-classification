package org.ml_methods_group.core.basic;

import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;

import java.util.List;

public class BasicSolutionDiff implements SolutionDiff {
    private final int sessionId;
    private final String before;
    private final String after;
    private final List<AtomicChange> changes;

    public BasicSolutionDiff(int sessionId, String before, String after, List<AtomicChange> changes) {
        this.sessionId = sessionId;
        this.before = before;
        this.after = after;
        this.changes = changes;
    }

    @Override
    public String getCodeBefore() {
        return before;
    }

    @Override
    public String getCodeAfter() {
        return after;
    }

    @Override
    public int getSessionId() {
        return sessionId;
    }

    @Override
    public List<AtomicChange> getChanges() {
        return changes;
    }
}

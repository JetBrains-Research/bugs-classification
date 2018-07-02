package org.ml_methods_group.database.proxy;

import org.ml_methods_group.core.Solution;
import org.ml_methods_group.core.Solution.Verdict;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.AtomicChange;

import java.util.List;

public class ProxyDiff implements SolutionDiff {

    private final Solution correct;
    private final Solution wrong;
    private final ProxyDatabase database;

    public ProxyDiff(int session, ProxyDatabase database) {
        this.correct = database.findBySession(session, Verdict.OK);
        this.wrong = database.findBySession(session, Verdict.FAIL);
        this.database = database;
    }

    @Override
    public Solution getWrongSolution() {
        return wrong;
    }

    @Override
    public Solution getCorrectSolution() {
        return correct;
    }

    @Override
    public List<AtomicChange> getChanges() {
        return database.getChanges(correct.getSessionId());
    }
}

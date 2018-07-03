package org.ml_methods_group.core;

import org.ml_methods_group.core.changes.AtomicChange;

import java.util.List;

public interface SolutionDiff {
    String getCodeBefore();
    String getCodeAfter();
    int getSessionId();
    List<AtomicChange> getChanges();
}

package org.ml_methods_group.database.proxy;

import org.ml_methods_group.core.Solution;
import org.ml_methods_group.core.Solution.Verdict;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.*;
import org.ml_methods_group.database.DatabaseException;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.primitives.Table;
import org.ml_methods_group.database.primitives.Tables;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import static org.ml_methods_group.core.changes.AtomicChange.ChangeType.*;

public class ProxySolutionDatabase implements SolutionDatabase {

    private final Database database;
    private final Table codes;
    private final Table diffs;

    public ProxySolutionDatabase() {
        this(new Database());
    }

    public ProxySolutionDatabase(Database database) {
        this.database = database;
        this.codes = database.getTable(Tables.CODES_HEADER);
        this.diffs = database.getTable(Tables.DIFF_HEADER);
    }

    private String getCode(int sessionId, Verdict verdict) {
        return codes.findFirst("session_id", sessionId, "verdict", verdict.ordinal())
                .map(wrapper -> wrapper.getStringValue("code"))
                .orElseThrow(NoSuchElementException::new);
    }

    private int getProblemId(int sessionId) {
        return codes.findFirst("session_id", sessionId)
                .map(wrapper -> wrapper.getIntValue("problem"))
                .orElseThrow(NoSuchElementException::new);
    }

    private List<AtomicChange> getChanges(int sessionId) {
        final List<AtomicChange> result = new ArrayList<>();
        diffs.find("session_id", sessionId)
                .forEachRemaining(wrapper -> result.add(fromResultWrapper(wrapper)));
        return result;
    }

    private AtomicChange fromResultWrapper(Table.ResultWrapper wrapper) {
        final int actionType = wrapper.getIntValue("action_type");
        if (actionType == DELETE.ordinal()) {
            return new DeleteChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("old_parent_type"),
                    wrapper.getIntValue("old_parent_of_parent_type"));
        } else if (actionType == INSERT.ordinal()) {
            return new InsertChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("parent_type"),
                    wrapper.getIntValue("parent_of_parent_type"));
        } else if (actionType == MOVE.ordinal()) {
            return new MoveChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("parent_type"),
                    wrapper.getIntValue("parent_of_parent_type"),
                    wrapper.getIntValue("old_parent_type"),
                    wrapper.getIntValue("old_parent_of_parent_type")
            );
        } else if (actionType == UPDATE.ordinal()) {
            return new UpdateChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("parent_type"),
                    wrapper.getIntValue("parent_of_parent_type"),
                    wrapper.getStringValue("old_label"));
        }
        throw new DatabaseException("Unexpected action type: " + actionType);
    }

    @Override
    public Iterator<AtomicChange> iterateChanges() {
        return transformToAtomicChange(diffs.listAll());
    }

    @Override
    public void close() throws Exception {
        database.close();
    }

    @Override
    public Solution findBySession(int sessionId, Verdict verdict) {
        return new ProxySolution(sessionId, verdict);
    }

    @Override
    public List<SolutionDiff> findByProblem(int problemId) {
        final List<SolutionDiff> result = new ArrayList<>();
        transformToSolutionDiff(codes.find("problem_id", problemId, "verdict", Verdict.OK.ordinal()))
            .forEachRemaining(result::add);
        return result;
    }

    @Override
    public List<SolutionDiff> getDiffs() {
        final List<SolutionDiff> result = new ArrayList<>();
        forEach(result::add);
        return result;
    }

    @Override
    public SolutionDiff getDiff(int session) {
        return new ProxySolutionDiff(session);
    }

    @Override
    public void insertSolution(Solution solution) {
        codes.insert(new Object[]{
                solution.getSessionId(),
                solution.getVerdict().ordinal(),
                solution.getCode(),
                solution.getProblemId()});
    }

    @Override
    public void clear() {
        database.dropTable(Tables.CODES_HEADER);
        database.dropTable(Tables.DIFF_HEADER);
        ;
    }

    @Override
    public void create() {
        database.createTable(Tables.CODES_HEADER);
        database.createTable(Tables.DIFF_HEADER);
    }

    @Override
    public Iterator<SolutionDiff> iterator() {
        return transformToSolutionDiff(codes.find("verdict", Verdict.OK.ordinal()));
    }

    @Override
    public void insertSolutionDiff(SolutionDiff solutionDiff) {
        for (AtomicChange change : solutionDiff.getChanges()) {
            diffs.insert(new Object[]{
                    solutionDiff.getSessionId(),
                    change.getChangeType().ordinal(),
                    change.getNodeType(),
                    change.getParentType(),
                    change.getParentOfParentType(),
                    change.getLabel(),
                    change.getOldParentType(),
                    change.getOldParentOfParentType(),
                    change.getOldLabel()
            });
        }
    }

    private Iterator<SolutionDiff> transformToSolutionDiff(Iterator<Table.ResultWrapper> iterator) {
        return new Iterator<SolutionDiff>() {
            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public SolutionDiff next() {
                return new ProxySolutionDiff(iterator.next().getIntValue("session_id"));
            }
        };
    }

    private Iterator<AtomicChange> transformToAtomicChange(Iterator<Table.ResultWrapper> iterator) {
        return new Iterator<AtomicChange>() {
            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public AtomicChange next() {
                return fromResultWrapper(iterator.next());
            }
        };
    }

    private class ProxySolution implements Solution {

        private final int sessionId;
        private final Verdict verdict;

        private ProxySolution(int sessionId, Verdict verdict) {
            this.sessionId = sessionId;
            this.verdict = verdict;
        }

        @Override
        public String getCode() {
            return ProxySolutionDatabase.this.getCode(sessionId, verdict);
        }

        @Override
        public int getProblemId() {
            return ProxySolutionDatabase.this.getProblemId(sessionId);
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

    private class ProxySolutionDiff implements SolutionDiff {

        private final int sessionId;

        private ProxySolutionDiff(int sessionId) {
            this.sessionId = sessionId;
        }

        @Override
        public String getCodeBefore() {
            return getCode(sessionId, Verdict.FAIL);
        }

        @Override
        public String getCodeAfter() {
            return getCode(sessionId, Verdict.OK);
        }

        @Override
        public int getSessionId() {
            return sessionId;
        }

        @Override
        public List<AtomicChange> getChanges() {
            return ProxySolutionDatabase.this.getChanges(sessionId);
        }
    }
}

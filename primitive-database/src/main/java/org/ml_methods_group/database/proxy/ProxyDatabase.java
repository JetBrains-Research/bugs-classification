package org.ml_methods_group.database.proxy;

import org.ml_methods_group.core.Solution;
import org.ml_methods_group.core.Solution.Verdict;
import org.ml_methods_group.core.SolutionDatabase;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.changes.*;
import org.ml_methods_group.database.Utils;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.primitives.Table;
import org.ml_methods_group.database.primitives.Tables;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;

import static org.ml_methods_group.core.Solution.Verdict.FAIL;
import static org.ml_methods_group.core.Solution.Verdict.OK;
import static org.ml_methods_group.core.changes.AtomicChange.ChangeType.*;

public class ProxyDatabase implements SolutionDatabase {

    private final Database database;

    public ProxyDatabase(Database database) {
        this.database = database;
    }

    @Override
    public Optional<String> getProblem(int problemId) {
        return Optional.empty();//todo
    }

    @Override
    public Solution findBySession(int sessionId, Verdict verdict) {
        final Table codes = database.getTable(Tables.codes_header);
        return new ProxySolution(codes, sessionId, verdict);
    }

    @Override
    public List<SolutionDiff> findByProblem(int problem) {
        try {
            final Table codes = database.getTable(Tables.codes_header);
            final List<SolutionDiff> result = new ArrayList<>();
            final Iterator<Table.ResultWrapper> iterator = codes.find("problem", problem);
            while (iterator.hasNext()) {
                final String id = iterator.next().getStringValue("id");
                if (id.endsWith("_0")) {
                    continue;
                }
                final int session = Integer.parseInt(id.substring(0, id.length() - 2));
                result.add(getDiff(session));
            }
            return result;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<SolutionDiff> getDiffs() {
        try {
            final Table codes = database.getTable(Tables.codes_header);
            final List<SolutionDiff> result = new ArrayList<>();
            final Iterator<Table.ResultWrapper> iterator = codes.listAll();
            while (iterator.hasNext()) {
                final String id = iterator.next().getStringValue("id");
                if (id.endsWith("_0")) {
                    continue;
                }
                final int session = Integer.parseInt(id.substring(0, id.length() - 2));
                result.add(getDiff(session));
            }
            return result;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public SolutionDiff getDiff(int session) {
        return new ProxyDiff(session,this);
    }

    @Override
    public void insertSolution(Solution solution) {
        try {
            final Table codes = database.getTable(Tables.codes_header);
            codes.insert(new Object[]{
                    solution.getSessionId() + (solution.getVerdict() == FAIL ? "_0" : "_1"),
                    solution.getCode(),
                    solution.getProblemId()});
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void clear() {
        try {
            database.dropTable(Tables.codes_header);
            database.dropTable(Tables.diff_header);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void create() {
        try {
            database.createTable(Tables.codes_header);
            database.createTable(Tables.diff_header);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    List<AtomicChange> getChanges(int session) {
        try {
            final Table diffs = database.getTable(Tables.diff_header);
            final Iterator<Table.ResultWrapper> iterator = diffs.find("session_id", session);
            if (!iterator.hasNext()) {
                return calculateChanges(session);
            }
            final List<AtomicChange> changes = new ArrayList<>();
            while (iterator.hasNext()) {
                final Table.ResultWrapper wrapper = iterator.next();
                changes.add(parse(wrapper));
            }
            return changes;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private List<AtomicChange> calculateChanges(int session) {
        try {
            final Table diffs = database.getTable(Tables.diff_header);
            final String before = findBySession(session, FAIL).getCode();
            final String after = findBySession(session, OK).getCode();
            final List<AtomicChange> changes = Utils.calculateChanges(before, after);
            for (AtomicChange change : changes) {
                diffs.insert(new Object[]{
                        session,
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
            return changes;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static AtomicChange parse(Table.ResultWrapper wrapper) throws SQLException {
        final int type = wrapper.getIntValue("action_type");
        if (type == DELETE.ordinal()) {
            return new DeleteChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("old_parent"),
                    wrapper.getIntValue("old_parent_of_parent"));
        } else if (type == INSERT.ordinal()) {
            return new InsertChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("parent_type"),
                    wrapper.getIntValue("parent_of_parent_type"));
        } else if (type == MOVE.ordinal()) {
            return new MoveChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("parent_type"),
                    wrapper.getIntValue("parent_of_parent_type"),
                    wrapper.getIntValue("old_parent"),
                    wrapper.getIntValue("old_parent_of_parent"));
        } else if (type == UPDATE.ordinal()) {
            return new UpdateChange(
                    wrapper.getIntValue("node_type"),
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("parent_type"),
                    wrapper.getIntValue("parent_of_parent_type"),
                    wrapper.getStringValue("old_label"));
        }
        throw new RuntimeException("Unexpected action type!");
    }
}

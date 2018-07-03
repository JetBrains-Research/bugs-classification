import org.ml_methods_group.classification.KNearestNeighbors;
import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.*;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.preparation.ChangesBuilder;
import org.ml_methods_group.core.vectorization.NearestSolutionVectorizer;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.core.vectorization.Wrapper;
import org.ml_methods_group.database.BasicIndexStorage;
import org.ml_methods_group.database.proxy.ProxySolutionDatabase;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.Utils.defaultStrategies;

public class Main {

    private static final int PROBLEM = 55673;

    public static void main(String[] args) throws SQLException, IOException {
        ProxySolutionDatabase database = new ProxySolutionDatabase();
        BasicIndexStorage storage = new BasicIndexStorage();
//        Utils.loadDatabase(Main.class.getResourceAsStream("/dataset.csv"), database, PROBLEM);
        final String ids = Utils.indexLabels(database, storage, 50, 100000);
        final Map<String, Long> dictionary = storage.loadIndex(ids);
        VectorTemplate<AtomicChange> template = Utils.generateTemplate(database, storage, defaultStrategies(dictionary),
                "index1", VectorTemplate.BasePostprocessors.RELATIVE, 10, Integer.MAX_VALUE);
        final List<SolutionDiff> solutions = database.getDiffs();
        System.out.println(solutions.size());
        Collections.shuffle(solutions, new Random(239566));
        final List<SolutionDiff> train = solutions.subList(0, 1000);
        final List<SolutionDiff> test = solutions.subList(1000, solutions.size());
        final List<Wrapper> wrappers = train.stream()
                .map(diff -> new Wrapper(template.process(diff.getChanges()), diff.getSessionId()))
                .collect(Collectors.toList());
        final Clusterer<Wrapper> clusterer = new HAC<>(0.2, 30, Wrapper::euclideanDistance);
        final List<List<Wrapper>> lists = clusterer.buildClusters(wrappers);
        final Map<Wrapper, Integer> mapping = new HashMap<>();
        for (int i = 0; i < lists.size(); i++) {
            for (Wrapper wrapper : lists.get(i)) {
                mapping.put(wrapper, i);
            }
        }
        final Classifier<Wrapper> classifier = new KNearestNeighbors(5);
        classifier.train(mapping);
        final NearestSolutionVectorizer vectorizer = new NearestSolutionVectorizer(train, template,
                new ChangesBuilder());
        final HintGenerator generator = new HintGenerator(vectorizer, classifier);
        for (int i = 0; i < lists.size(); i++) {
            generator.setHint(i, "hint#" + i);
        }
        System.out.println(generator.getTip(test.get(0).getCodeBefore()));
    }
}

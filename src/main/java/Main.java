import org.ml_methods_group.classification.KNearestNeighbors;
import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.*;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.preparation.ChangesBuilder;
import org.ml_methods_group.core.selection.CenterSelector;
import org.ml_methods_group.core.vectorization.NearestSolutionVectorizer;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.core.vectorization.Wrapper;
import org.ml_methods_group.database.BasicIndexStorage;
import org.ml_methods_group.database.proxy.ProxySolutionDatabase;
import org.ml_methods_group.ui.ConsoleIO;
import org.ml_methods_group.ui.UtilsUI;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.Utils.defaultStrategies;
import static org.ml_methods_group.core.selection.CenterSelector.Mode.MAX;

public class Main {

    private static final int PROBLEM = 55673;

    public static void main(String[] args) throws SQLException, IOException {
        PrintWriter out = new PrintWriter("out.csv");
        ProxySolutionDatabase database = new ProxySolutionDatabase();
        BasicIndexStorage storage = new BasicIndexStorage();
//        UtilsUI.loadDatabase(Main.class.getResourceAsStream("/dataset.csv"), database, PROBLEM);
        final String ids = Utils.indexLabels(database, storage, 50, 100000);
        final Map<String, Long> dictionary = storage.loadIndex(ids);
        VectorTemplate<AtomicChange> template = Utils.generateTemplate(database, storage, defaultStrategies(dictionary),
                "index1", VectorTemplate.BasePostprocessors.STANDARD, 2, Integer.MAX_VALUE);
        final List<SolutionDiff> solutions = database.getDiffs();
        System.out.println(solutions.size());
        Collections.shuffle(solutions, new Random(239566));
        final List<SolutionDiff> train = solutions.subList(0, 1000);
        final List<SolutionDiff> test = solutions.subList(1000, solutions.size());
        final List<Wrapper> wrappers = train.stream()
                .map(diff -> new Wrapper(template.process(diff.getChanges()), diff.getSessionId()))
                .collect(Collectors.toList());
        final List<List<Wrapper>> lists;
        try (HAC<Wrapper> clusterer = new HAC<>(0.2, 30, Wrapper::squaredEuclideanDistance)) {
            lists = clusterer.buildClusters(wrappers);
        }
        final Map<Wrapper, Integer>mapping=new HashMap<>() ;
        for (int i = 0; i < lists.size(); i++) {
            out.write(lists.get(i).size() + "\n");
            if (lists.get(i).size() > 10 && lists.get(i).size() < 20) {
                writeCluster(lists.get(i), template);
            }
        }
        out.close();
        final Classifier<Wrapper> classifier = new KNearestNeighbors(5);
        classifier.train(mapping);
        final NearestSolutionVectorizer vectorizer = new NearestSolutionVectorizer(train, template,
                new ChangesBuilder());
        ConsoleIO console = new ConsoleIO();
        final CenterSelector<Wrapper> selector = new CenterSelector<>(Wrapper::squaredEuclideanDistance, 0.2, MAX);
        final List<String> marks = UtilsUI.markClusters(lists, selector, database, console, l -> l.size() > 5);
        final HintGenerator generator = new HintGenerator(vectorizer, classifier);
        for (int i = 0; i < marks.size(); i++) {
            generator.setHint(i, marks.get(i));
        }
        console.write("Start testing:");
        for (int i = 0; i < 20; i++) {
            final SolutionDiff example = test.get(i);
            console.write("------------------------next-case-----------------------------");
            console.write(example);
            console.write(generator.getTip(example.getCodeBefore()));
            console.readLine();
        }
    }

    public static void writeCluster(List<Wrapper> wrappers, VectorTemplate<AtomicChange> template) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter("vectors.csv");
        for (int i = 0; i < wrappers.get(0).vector.length; i++) {
            final int index = i;
            final String line = wrappers.stream()
                    .mapToDouble(wrapper -> wrapper.vector[index])
                    .mapToObj(Double::toString)
                    .collect(Collectors.joining(","));
            writer.println(line);
        }
        writer.close();
    }
}

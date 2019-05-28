package org.ml_methods_group.classification.classifiers;

import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.MarkedClusters;
import org.ml_methods_group.common.Wrapper;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class WekaClassifier<T> implements Classifier<Wrapper<double[], T>, Boolean> {

    private static final Attribute CLASS_ATTRIBUTE = createBooleanAttribute("mark");

    private final weka.classifiers.Classifier classifier;
    private final Instances template;
    private final int featuresSize;

    public WekaClassifier(weka.classifiers.Classifier classifier, int featuresSize) {
        this.classifier = classifier;
        this.featuresSize = featuresSize;
        template = createInstances(featuresSize);
    }

    @Override
    public void train(MarkedClusters<Wrapper<double[], T>, Boolean> samples) {
        final Map<Wrapper<double[], T>, Boolean> marks = samples.getFlatMarks();
        final int datasetSize = marks.size();
        final Instances dataset = new Instances(template, datasetSize);
        for (Map.Entry<Wrapper<double[], T>, Boolean> entry : marks.entrySet()) {
            final double[] features = entry.getKey().getFeatures();
            final double[] attributes = new double[featuresSize + 1];
            System.arraycopy(features, 0, attributes, 0, featuresSize);
            attributes[featuresSize] = entry.getValue() ? 1.0 : 0.0;
            dataset.add(new Instance(1.0, attributes));
        }
        try {
            classifier.buildClassifier(dataset);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<Boolean, Double> reliability(Wrapper<double[], T> value) {
        final double[] features = value.getFeatures();
        final double[] attributes = new double[featuresSize + 1];
        System.arraycopy(features, 0, attributes, 0, featuresSize);
        Instance instance = new Instance(1.0, attributes);
        instance.setDataset(template);
        final HashMap<Boolean, Double> results = new HashMap<>();
        try {
            final double[] probabilities = classifier.distributionForInstance(instance);
            results.put(true, probabilities[1]);
            results.put(false, probabilities[0]);
        } catch (Exception e) {
            results.put(true, 0d);
            results.put(false, 0d);
        }
        return results;
    }

    private static Instances createInstances(int featuresSize) {
        final FastVector attributes = new FastVector(featuresSize + 1);
        for (int i = 0; i < featuresSize; i++) {
            attributes.addElement(new Attribute("f" + i));
        }
        attributes.addElement(CLASS_ATTRIBUTE.copy());
        final Instances instances = new Instances("dataset", attributes, 0);
        instances.setClass(instances.attribute(featuresSize));
        return instances;
    }

    private static Attribute createBooleanAttribute(String name) {
        final FastVector classValues = new FastVector(2);
        classValues.addElement("false");
        classValues.addElement("true");
        return new Attribute(name, classValues);
    }
}

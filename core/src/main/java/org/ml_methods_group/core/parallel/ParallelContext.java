package org.ml_methods_group.core.parallel;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class ParallelContext implements AutoCloseable {
    private final ExecutorService service;

    public ParallelContext() {
        this.service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    @Override
    public void close() {
        service.shutdown();
    }

    public <A, V> A runParallel(List<V> values, Supplier<A> accumulatorFactory,
                                 BiFunction<V, A, A> processor, BinaryOperator<A> combiner) {
        final List<Callable<A>> tasks = splitValues(values).stream()
                .sequential()
                .map(list -> new Task<>(list, accumulatorFactory, processor))
                .collect(Collectors.toList());
        final List<Future<A>> results = new ArrayList<>();
        for (Callable<A> task : tasks) {
            results.add(service.submit(task));
        }
        return results.stream()
                .sequential()
                .map(this::getResult)
                .reduce(combiner)
                .orElseGet(accumulatorFactory);
    }

    private <V> List<List<V>> splitValues(List<V> values) {
        final List<List<V>> lists = new ArrayList<>();
        final int valuesCount = values.size();
        final int blocksCount = Math.min(4, values.size());
        final int blockSize = (valuesCount - 1) / blocksCount + 1; // round up
        for (int blockStart = 0; blockStart < valuesCount; blockStart += blockSize) {
            lists.add(values.subList(blockStart, Math.min(blockStart + blockSize, valuesCount)));
        }
        return lists;
    }

    private <V> V getResult(Future<V> future) {
        while (true) {
            try {
                return future.get();
            } catch (InterruptedException ignored) {
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private class Task<A, V> implements Callable<A> {
        private final List<V> values;
        private final Supplier<A> accumulatorFactory;
        private final BiFunction<V, A, A> processor;

        private Task(List<V> values, Supplier<A> accumulatorFactory, BiFunction<V, A, A> processor) {
            this.values = values;
            this.accumulatorFactory = accumulatorFactory;
            this.processor = processor;
        }

        public A call() {
            A accumulator = accumulatorFactory.get();
            for (V value : values) {
                accumulator = processor.apply(value, accumulator);
            }
            return accumulator;
        }
    }
}

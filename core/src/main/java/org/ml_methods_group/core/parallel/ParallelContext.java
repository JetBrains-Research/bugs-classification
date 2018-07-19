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

    public <V, A> A runParallel(List<V> values,
                                Supplier<A> accumulatorFactory,
                                BiFunction<V, A, A> processor,
                                BinaryOperator<A> combiner) {
        return this.<V, Void, A>runParallel(values,
                accumulatorFactory,
                () -> null,
                (value, context, accumulator) -> processor.apply(value, accumulator),
                combiner);
    }

    public <V, C, A> A runParallel(List<V> values,
                                   Supplier<A> accumulatorFactory,
                                   C context,
                                   ParallelProcessor<V, C, A> processor,
                                   BinaryOperator<A> combiner) {
        return this.<V, C, A>runParallel(values,
                accumulatorFactory,
                () -> context,
                processor,
                combiner);
    }

    public <V, C, A> A runParallel(List<V> values,
                                   Supplier<A> accumulatorFactory,
                                   Supplier<C> contextFactory,
                                   ParallelProcessor<V, C, A> processor,
                                   BinaryOperator<A> combiner) {
        final List<Callable<A>> tasks = splitValues(values).stream()
                .sequential()
                .map(list -> new Task<>(list, accumulatorFactory, contextFactory, processor))
                .collect(Collectors.toList());
        final List<Future<A>> results = new ArrayList<>();
        for (Callable<A> task : tasks) {
            results.add(service.submit(task));
        }
        return results.stream()
                .sequential()
                .map(ParallelContext::getResult)
                .reduce(combiner)
                .orElseGet(accumulatorFactory);
    }

    private static <V> List<List<V>> splitValues(List<V> values) {
        final List<List<V>> lists = new ArrayList<>();
        final int valuesCount = values.size();
        final int blocksCount = Math.min(4, values.size());
        final int blockSize = (valuesCount - 1) / blocksCount + 1; // round up
        for (int blockStart = 0; blockStart < valuesCount; blockStart += blockSize) {
            lists.add(values.subList(blockStart, Math.min(blockStart + blockSize, valuesCount)));
        }
        return lists;
    }

    private static <V> V getResult(Future<V> future) {
        while (true) {
            try {
                return future.get();
            } catch (InterruptedException ignored) {
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private class Task<V, C, A> implements Callable<A> {
        private final List<V> values;
        private final Supplier<A> accumulatorFactory;
        private final Supplier<C> contextFactory;
        private final ParallelProcessor<V, C, A> processor;

        private Task(List<V> values, Supplier<A> accumulatorFactory, Supplier<C> contextFactory,
                     ParallelProcessor<V, C, A> processor) {
            this.values = values;
            this.accumulatorFactory = accumulatorFactory;
            this.contextFactory = contextFactory;
            this.processor = processor;
        }

        public A call() {
            final C context = contextFactory.get();
            A accumulator = accumulatorFactory.get();
            for (V value : values) {
                accumulator = processor.process(value, context, accumulator);
            }
            return accumulator;
        }
    }
}

package com.dubiouscandle.dubiousdl;

import java.util.Random;

public class Batch {
	private final Random random;
	private int index;

	private float[][] embeddings;
	private int[] labels;

	private Matrix input;
	private Matrix target;

	private final int batchSize;
	private final int numClasses;
	private final int embeddingSize;

	public Batch(float[][] embeddings, int[] labels, int len, int numClasses, int embeddingSize, int batchSize,
			Random random) {
		this.random = random;
		this.embeddingSize = embeddingSize;
		this.embeddings = new float[len][];

		for (int i = 0; i < len; i++) {
			this.embeddings[i] = embeddings[i].clone();
			if (embeddings[i].length != embeddingSize) {
				throw new IllegalArgumentException("Invalid embedding.");
			}
		}

		this.labels = labels.clone();

		input = new Matrix(embeddingSize, batchSize);
		target = new Matrix(numClasses, batchSize);

		this.batchSize = batchSize;
		this.numClasses = numClasses;
		index = -batchSize;
		shuffle();
		next();
	}

	public void next() {
		index += batchSize;

		if (index + batchSize > embeddings.length) {
			shuffle();
			index = 0;
		}

		for (int i = 0; i < batchSize; i++) {
			int label = labels[index + i];
			if (label < 0 || label >= numClasses) {
				throw new IllegalArgumentException("Label " + label + " out of bounds.");
			}

			for (int j = 0; j < embeddingSize; j++) {
				input.set(j, i, embeddings[index + i][j]);
			}
			for (int j = 0; j < numClasses; j++) {
				target.set(j, i, j == label ? 1.0f : 0.0f);
			}
		}
	}

	private void shuffle() {
		for (int i = embeddings.length - 1; i > 0; i--) {
			int j = random.nextInt(i + 1);

			swap(i, j);
		}
	}

	private void swap(int i, int j) {
		float[] tempValue = embeddings[i];
		embeddings[i] = embeddings[j];
		embeddings[j] = tempValue;

		int tempLabel = labels[i];
		labels[i] = labels[j];
		labels[j] = tempLabel;
	}

	public Matrix input() {
		return input;
	}

	public Matrix target() {
		return target;
	}

}

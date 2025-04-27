package com.dubiouscandle.dubiousdl;

import java.util.Random;

public class Batch {
	private final Random random;
	private int index;

	private float[][] values;
	private int[] labels;

	private Matrix input;
	private Matrix target;

	private final int batchSize;
	private final int numClasses;

	public Batch(float[][] values, int[] labels, int len, int numClasses, int batchSize, Random random) {
		this.random = random;
		this.values = new float[len][];

		for (int i = 0; i < len; i++) {
			this.values[i] = values[i].clone();
		}

		this.labels = new int[len];
		System.arraycopy(labels, 0, this.labels, 0, len);
		for (int label : this.labels) {
			if (label < 0 || label >= numClasses) {
				throw new IllegalArgumentException("Label " + label + " out of bounds.");
			}
		}

		input = new Matrix(numClasses, batchSize);
		target = new Matrix(numClasses, batchSize);

		this.batchSize = batchSize;
		this.numClasses = numClasses;
	}

	public void next() {
		index += batchSize;

		if (index + batchSize > values.length) {
			shuffle();
			index = 0;
		}

		for (int i = 0; i < batchSize; i++) {
			int label = labels[index + i];

			for (int j = 0; j < numClasses; j++) {
				input.set(i, j, values[index + i][j]);
				target.set(i, j, j == label ? 1.0f : 0.0f);
			}
		}
	}

	private void shuffle() {
		for (int i = values.length - 1; i > 0; i--) {
			int j = random.nextInt(i + 1);

			swap(i, j);
		}
	}

	private void swap(int i, int j) {
		float[] tempValue = values[i];
		values[i] = values[j];
		values[j] = tempValue;

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

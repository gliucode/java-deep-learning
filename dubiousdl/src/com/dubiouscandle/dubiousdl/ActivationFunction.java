package com.dubiouscandle.dubiousdl;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {

	ActivationFunction LEAKY_RELU = new ActivationFunction() {
		private static final long serialVersionUID = -4817516167369688053L;

		@Override
		public float getImpl(float x) {
			return x <= 0 ? 0.01f * x : x;
		}

		@Override
		public float derivImpl(float x) {
			return x <= 0 ? 0.01f : 1;
		}
	};

	ActivationFunction TANH = new ActivationFunction() {
		private static final long serialVersionUID = -8218100390371585663L;

		@Override
		public float getImpl(float x) {
			return (float) Math.tanh(x);
		}

		@Override
		public float derivImpl(float x) {
			float tanh = (float) Math.tanh(x);
			return 1 - tanh * tanh;
		}
	};

	ActivationFunction SOFT_PLUS = new ActivationFunction() {
		private static final long serialVersionUID = -3310120748805037640L;

		@Override
		public float getImpl(float x) {
			return (float) Math.log(1 + Math.exp(x));
		}

		@Override
		public float derivImpl(float x) {
			return (float) (1.0 / (1.0 + Math.exp(-x)));
		}
	};

	ActivationFunction SIN = new ActivationFunction() {
		private static final long serialVersionUID = 4514832160859070858L;

		@Override
		public float getImpl(float x) {
			return (float) Math.sin(x);
		}

		@Override
		public float derivImpl(float x) {
			return (float) Math.cos(x);
		}
	};

	ActivationFunction ELU = new ActivationFunction() {
		private static final long serialVersionUID = 6788758950796204326L;

		@Override
		public float getImpl(float x) {
			return x > 0 ? x : (float) (Math.exp(x) - 1);
		}

		@Override
		public float derivImpl(float x) {
			return x > 0 ? 1 : (float) Math.exp(x);
		}
	};

	ActivationFunction SWISH = new ActivationFunction() {
		private static final long serialVersionUID = 8660878089911283864L;

		@Override
		public float getImpl(float x) {
			return x / (float) (1 + Math.exp(-x));
		}

		@Override
		public float derivImpl(float x) {
			float sigmoid = (float) (1 / (1 + Math.exp(-x)));
			return sigmoid * (1 + x * (1 - sigmoid));
		}
	};

	ActivationFunction IDENTITY = new ActivationFunction() {
		private static final long serialVersionUID = -4817516167369688053L;

		@Override
		public float getImpl(float x) {
			return x;
		}

		@Override
		public float derivImpl(float x) {
			return 1;
		}
	};

	ActivationFunction RELU = new ActivationFunction() {
		private static final long serialVersionUID = 5354573844677938806L;

		@Override
		public float getImpl(float x) {
			return x <= 0 ? 0 : x;
		}

		@Override
		public float derivImpl(float x) {
			return x <= 0 ? 0 : 1;
		}
	};

	ActivationFunction SIGMOID = new ActivationFunction() {
		private static final long serialVersionUID = -1355151573296827751L;

		@Override
		public float getImpl(float x) {
			return (float) (1.0 / (1.0 + Math.exp(-x)));
		}

		@Override
		public float derivImpl(float x) {
			float sigmoid = (float) (1.0 / (1.0 + Math.exp(-x)));
			return sigmoid * (1 - sigmoid);
		}
	};

	/**
	 * Computes the activation for a single element.
	 * 
	 * @param x the input value
	 * @return the activated value
	 */
	float getImpl(float x);

	/**
	 * Computes the derivative of the activation for a single element.
	 * 
	 * @param x the input value
	 * @return the derivative of the activation function at the given input value
	 */
	float derivImpl(float x);

	/**
	 * Applies the activation function element-wise to an array of inputs and stores the results in the output array.
	 * 
	 * @param x   the input array
	 * @param out the output array to store results
	 */
	default void get(float[] x, float[] out) {
		if (x.length != out.length) {
			throw new IllegalArgumentException("Input and output arrays do not match in length.");
		}

		for (int i = 0; i < x.length; i++) {
			out[i] = getImpl(x[i]);
		}
	}

	/**
	 * Applies the derivative of the activation function element-wise to an array of inputs and stores the results in the output array.
	 * 
	 * @param x   the input array
	 * @param out the output array to store results
	 */
	default void deriv(float[] x, float[] out) {
		if (x.length != out.length) {
			throw new IllegalArgumentException("Input and output arrays do not match in length.");
		}

		for (int i = 0; i < x.length; i++) {
			out[i] = derivImpl(x[i]);
		}
	}

	/*
	 * computes softmax in place on the array
	 */
	static void softmax(float[] arr) {
		float max = Float.NEGATIVE_INFINITY;

		for (float x : arr) {
			if (x > max) {
				max = x;
			}
		}

		float sum = 0;
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (float) Math.exp(arr[i] - max);
			sum += arr[i];
		}

		for (int i = 0; i < arr.length; i++) {
			arr[i] /= sum;
		}
	}
}

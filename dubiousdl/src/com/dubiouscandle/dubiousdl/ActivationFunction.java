package com.dubiouscandle.dubiousdl;

import java.io.Serializable;
import java.util.Arrays;

public interface ActivationFunction extends Serializable {
	ActivationFunction LEAKY_RELU = new ActivationFunction() {
		private static final long serialVersionUID = -4817516167369688053L;

		@Override
		public void getImpl(float[] x, float[] out) {
			for (int i = 0; i < x.length; i++) {
				out[i] = x[i] <= 0 ? 0.01f * x[i] : x[i];
			}
		}

		@Override
		public void derivImpl(float[] x, float[] out) {
			for (int i = 0; i < x.length; i++) {
				out[i] = x[i] <= 0 ? 0.01f : 1;
			}
		}
	};
	ActivationFunction IDENTITY = new ActivationFunction() {
		private static final long serialVersionUID = -4817516167369688053L;

		@Override
		public void getImpl(float[] x, float[] out) {
			System.arraycopy(x, 0, out, 0, x.length);
		}

		@Override
		public void derivImpl(float[] x, float[] out) {
			Arrays.fill(out, 1);
		}
	};
	ActivationFunction RELU = new ActivationFunction() {
		private static final long serialVersionUID = 5354573844677938806L;

		@Override
		public void getImpl(float[] x, float[] out) {
			for (int i = 0; i < x.length; i++) {
				out[i] = x[i] <= 0 ? 0 : x[i];
			}
		}

		@Override
		public void derivImpl(float[] x, float[] out) {
			for (int i = 0; i < x.length; i++) {
				out[i] = x[i] <= 0 ? 0 : 1;
			}
		}
	};
	ActivationFunction SIGMOID = new ActivationFunction() {
		private static final long serialVersionUID = -1355151573296827751L;

		@Override
		public void getImpl(float[] x, float[] out) {
			for (int i = 0; i < x.length; i++) {
				out[i] = (float) (1.0 / (1.0 + Math.exp(-x[i])));
			}
		}

		@Override
		public void derivImpl(float[] x, float[] out) {
			for (int i = 0; i < x.length; i++) {
				float y = (float) (1.0 / (1.0 + Math.exp(-x[i])));
				out[i] = y * (1 - y);
			}
		}
	};

	/**
	 * Computes the activation for every single element.
	 * 
	 * @param mat
	 * @param derivOut
	 */
	void getImpl(float[] x, float[] out);

	/**
	 * 
	 * @param matrix
	 * @param out
	 */
	void derivImpl(float[] x, float[] out);

	default void deriv(float[] x, float[] out) {
		if (x.length != out.length) {
			throw new IllegalArgumentException("Input and output arrays do not match in length.");
		}

		derivImpl(x, out);
	}

	default void get(float[] x, float[] out) {
		if (x.length != out.length) {
			throw new IllegalArgumentException("Input and output arrays do not match in length.");
		}

		getImpl(x, out);
	}
}

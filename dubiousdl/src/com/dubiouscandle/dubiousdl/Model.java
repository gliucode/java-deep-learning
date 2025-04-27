package com.dubiouscandle.dubiousdl;

import java.io.Serializable;

public class Model implements Serializable {
	private static final long serialVersionUID = -990276425379370768L;
	protected final int input_size;
	protected final int output_size;
	protected final int m;
	protected final int L;

	protected final Matrix[] W;
	protected final Matrix[] a;
	protected final Matrix[] z;
	protected final Matrix[] b;
	protected final int[] n;
	protected final ActivationFunction[] g;

	public Model(ActivationFunction[] g, int[] layerSizes, int m, Initializer initializer) {
		if (g.length + 1 != layerSizes.length) {
			throw new IllegalArgumentException("Invalid activation functions length or layer sizes length.");
		}

		n = layerSizes.clone();

		this.m = m;
		this.input_size = n[0];
		this.output_size = n[n.length - 1];
		L = n.length - 1;

		this.g = new ActivationFunction[L + 1];
		System.arraycopy(g, 0, this.g, 1, g.length);

		a = new Matrix[L + 1];
		z = new Matrix[L + 1];
		W = new Matrix[L + 1];
		b = new Matrix[L + 1];

		a[0] = new Matrix(n[0], m);
		z[0] = new Matrix(n[0], m);
		b[0] = new Matrix(n[0], 1);

		for (int l = 1; l <= L; l++) {
			a[l] = new Matrix(n[l], m);
			z[l] = new Matrix(n[l], m);
			W[l] = new Matrix(n[l], n[l - 1]);
			b[l] = new Matrix(n[l], 1);
		}
		initializeWeights(initializer);
	}

	private void initializeWeights(Initializer initializer) {
		for (int l = 1; l <= L; l++) {
			int fan_in = W[L].cols();
			int fan_out = W[L].rows();

			for (int i = 0; i < W[l].data().length; i++) {
				W[l].data()[i] = initializer.get(fan_in, fan_out);
			}
		}
	}

	public Model(Model model, int m) {
		this.m = m;
		this.input_size = model.input_size;
		this.output_size = model.output_size;
		this.n = model.n.clone();

		L = model.L;

		a = new Matrix[L + 1];
		z = new Matrix[L + 1];
		W = new Matrix[L + 1];
		b = new Matrix[L + 1];

		int n0 = model.a[0].rows();
		a[0] = new Matrix(n0, m);
		z[0] = new Matrix(n0, m);
		b[0] = new Matrix(n0, 1);

		for (int l = 1; l <= L; l++) {
			int n = model.a[l].rows();
			a[l] = new Matrix(n, m);
			z[l] = new Matrix(n, m);
			W[l] = new Matrix(model.W[l]);
			b[l] = new Matrix(model.b[l]);
		}

		this.g = model.g;
	}

	public void forwardPropagate(Matrix in, Matrix out) {
		if (in.rows() != input_size || in.cols() != m) {
			throw new IllegalArgumentException("Invalid input dimensions.");
		}
		if (out.rows() != output_size || out.cols() != m) {
			throw new IllegalArgumentException("Invalid output dimensions.");
		}

		a[0].set(in);

		for (int l = 1; l <= L; l++) {
			computeForwardPropagationStep(l);
		}

		out.set(a[L]);
	}

	/**
	 * forward propagates into the next layer using the data from layer
	 * 
	 * @param layer
	 */
	private void computeForwardPropagationStep(int l) {
		Matrix.dot(W[l], a[l - 1], z[l]);
		z[l].addBroadcasted(b[l]);
		g[l].get(z[l].data(), a[l].data());
	}
}

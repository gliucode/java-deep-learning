package com.dubiouscandle.dubiousdl;

public class Adam {
	private static final float EPSILON = 1e-5f;

	private final Model model;

	// cached matrices to avoid reallocation every iteration
	private final Matrix[] da;
	private final Matrix[] dz;
	private final Matrix[] dW;
	private final Matrix[] db;
	public final Matrix output;

	protected final Matrix[] mW;
	protected final Matrix[] vW;
	protected final Matrix[] mb;
	protected final Matrix[] vb;

	protected final float B1, B2;

	private final int L;
	private final Matrix[] W;
	private final Matrix[] a;
	private final Matrix[] Z;
	private final Matrix[] b;
	private final ActivationFunction[] g;

	private final LossFunction J;

	private int t = 1;

	public Adam(Model model, LossFunction J, float B1, float B2) {
		this.model = model;
		this.B1 = B1;
		this.B2 = B2;

		this.J = J;

		output = new Matrix(model.output_size, model.m);

		this.L = model.L;
		this.W = model.W;
		this.a = model.a;
		this.Z = model.z;
		this.b = model.b;

		this.g = model.g;

		dW = Matrix.deepClone(model.W);
		db = Matrix.deepClone(model.b);

		da = Matrix.deepClone(model.a);
		dz = Matrix.deepClone(model.z);

		mW = new Matrix[L + 1];
		vW = new Matrix[L + 1];
		mb = new Matrix[L + 1];
		vb = new Matrix[L + 1];

		for (int l = 1; l <= L; l++) {
			mW[l] = Matrix.emptyCopyOf(W[l]);
			vW[l] = Matrix.emptyCopyOf(W[l]);
			mb[l] = Matrix.emptyCopyOf(b[l]);
			vb[l] = Matrix.emptyCopyOf(b[l]);
		}
	}

	public void step(Matrix input, Matrix target, float alpha) {
		if (target.rows() != output.rows() || target.cols() != output.cols()) {
			throw new IllegalArgumentException("Invalid expected output.");
		}
		if (input.rows() != model.input_size || input.cols() != model.m) {
			throw new IllegalArgumentException("Invalid expected output.");
		}

		clearCache();
		model.forwardPropagate(input, output);

		J.getError(output, target, dz[L]);

		for (int l = L; l >= 1; l--) {
			computeBackPropagationStep(l, alpha);
		}

		t++;
	}

	private void clearCache() {
		for (Matrix matrix : da) {
			if (matrix != null)
				matrix.clear();
		}
		for (Matrix matrix : dz) {
			if (matrix != null)
				matrix.clear();
		}
		for (Matrix matrix : dW) {
			if (matrix != null)
				matrix.clear();
		}
		for (Matrix matrix : db) {
			if (matrix != null)
				matrix.clear();
		}
		output.clear();
	}

	private void computeBackPropagationStep(int l, float alpha) {
		if (l != L) {
			g[l].deriv(Z[l].data(), dz[l].data());
			Matrix.hadamardProduct(dz[l], da[l]);
		}

		a[l - 1].transpose();
		Matrix.multiply(dz[l], a[l - 1], dW[l]);
		a[l - 1].transpose();
		dW[l].multiply(1.0f / model.m);

		dz[l].sumOfCols(db[l]);
		db[l].multiply(1.0f / model.m);

		W[l].transpose();
		Matrix.multiply(W[l], dz[l], da[l - 1]);
		W[l].transpose();

		for (int i = 0; i < mW[l].data().length; i++) {
			float m = mW[l].data()[i];
			float v = vW[l].data()[i];
			float g = dW[l].data()[i];

			m = B1 * m + (1 - B1) * g;
			v = B2 * v + (1 - B2) * g * g;

			float m_hat = m / (float) (1 - Math.pow(B1, t));
			float v_hat = v / (float) (1 - Math.pow(B2, t));

			W[l].data()[i] = W[l].data()[i] - alpha * m_hat / (float) Math.sqrt(v_hat + EPSILON);

			mW[l].data()[i] = m;
			vW[l].data()[i] = v;
		}
		
		for (int i = 0; i < mb[l].data().length; i++) {
			float m = mb[l].data()[i];
			float v = vb[l].data()[i];
			float g = db[l].data()[i];

			m = B1 * m + (1 - B1) * g;
			v = B2 * v + (1 - B2) * g * g;

			float m_hat = m / (float) (1 - Math.pow(B1, t));
			float v_hat = v / (float) (1 - Math.pow(B2, t));

			b[l].data()[i] = b[l].data()[i] - alpha * m_hat / (float) Math.sqrt(v_hat + EPSILON);

			mb[l].data()[i] = m;
			vb[l].data()[i] = v;
		}
	}

}

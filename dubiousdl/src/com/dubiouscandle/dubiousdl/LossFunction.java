package com.dubiouscandle.dubiousdl;

public interface LossFunction {
	/**
	 * note: if you use this loss function the output layers activation should be IDENTITY
	 */
	LossFunction SOFT_MAX_CROSS_ENTROPY_LOSS = new LossFunction() {
		@Override
		public void getErrorImpl(float[] output, float[] target, float[] out) {
			int numClasses = output.length;

			float[] softmaxOutput = new float[numClasses];
			float maxOutput = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < numClasses; i++) {
				maxOutput = Math.max(maxOutput, output[i]);
			}

			float sumExp = 0.0f;
			for (int i = 0; i < numClasses; i++) {
				sumExp += Math.exp(output[i] - maxOutput);
			}
			for (int i = 0; i < numClasses; i++) {
				softmaxOutput[i] = (float) Math.exp(output[i] - maxOutput) / sumExp;
			}
			for (int i = 0; i < numClasses; i++) {
				out[i] = softmaxOutput[i] - target[i];
			}
		}

	    @Override
	    public float getLossImpl(float[] output, float[] target) {
	        int numClasses = output.length;
	        float loss = 0.0f;

	        float maxOutput = Float.NEGATIVE_INFINITY;
	        for (int i = 0; i < numClasses; i++) {
	            maxOutput = Math.max(maxOutput, output[i]);
	        }

	        float sumExp = 0.0f;
	        for (int i = 0; i < numClasses; i++) {
	            sumExp += Math.exp(output[i] - maxOutput);
	        }

	        float[] softmaxOutput = new float[numClasses];
	        for (int i = 0; i < numClasses; i++) {
	            softmaxOutput[i] = (float) Math.exp(output[i] - maxOutput) / sumExp;
	        }

	        for (int i = 0; i < numClasses; i++) {
	            if (target[i] == 1) {
	                loss -= Math.log(softmaxOutput[i]);
	            }
	        }

	        return loss;
	    }

	};

	/**
	 * note: if you use this loss function the output layers activation should be IDENTITY
	 */
	LossFunction BINARY_CROSS_ENTROPY_LOSS = new LossFunction() {

		private static final float EPSILON = 1e-7f;

		@Override
		public void getErrorImpl(float[] output, float[] target, float[] out) {
			float z = output[0];
			float y = target[0];

			float yHat = (float) (1.0 / (1.0 + Math.exp(-z)));
			out[0] = yHat - y;
		}

		@Override
		public float getLossImpl(float[] output, float[] target) {
			float z = output[0];
			float y = target[0];

			float yHat = (float) (1.0 / (1.0 + Math.exp(-z)));

			if (yHat < EPSILON) {
				yHat = EPSILON;
			} else if (yHat > 1 - EPSILON) {
				yHat = 1 - EPSILON;
			}

			return (float) (-(y * Math.log(yHat) + (1 - y) * Math.log(1 - yHat)));
		}
	};

	void getErrorImpl(float[] output, float[] target, float[] out);

	float getLossImpl(float[] output, float[] target);

	/**
	 * computes the error between output and target and stores it in error
	 * 
	 * @param output
	 * @param target
	 * @param error
	 */
	default void getError(Matrix output, Matrix target, Matrix error) {
		int numClasses = output.rows();
		int batchSize = output.cols();

		float[] outputArr = new float[numClasses];
		float[] targetArr = new float[numClasses];
		float[] errorArr = new float[numClasses];

		for (int batch = 0; batch < batchSize; batch++) {
			for (int i = 0; i < numClasses; i++) {
				outputArr[i] = output.get(i, batch);
				targetArr[i] = target.get(i, batch);
			}
			getErrorImpl(outputArr, targetArr, errorArr);
			for (int j = 0; j < output.rows(); j++) {
				error.set(j, batch, errorArr[j]);
			}
		}
	}

	/**
	 * computes the average loss between output and target
	 * 
	 * @param output
	 * @param target
	 * @return the average loss
	 */
	default float getLoss(Matrix output, Matrix target) {
		if (output.rows() != target.rows() || output.cols() != target.cols()) {
			throw new IllegalArgumentException("Dimension mismatch");
		}

		int numClasses = output.rows();
		int batchSize = output.cols();

		float[] outputArr = new float[numClasses];
		float[] targetArr = new float[numClasses];

		float sum = 0;
		for (int batch = 0; batch < batchSize; batch++) {
			for (int i = 0; i < numClasses; i++) {
				outputArr[i] = output.get(i, batch);
				targetArr[i] = target.get(i, batch);
			}
			sum += getLossImpl(outputArr, targetArr);
		}

		return sum / batchSize;
	}
}

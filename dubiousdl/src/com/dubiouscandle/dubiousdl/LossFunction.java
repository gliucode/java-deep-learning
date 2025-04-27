package com.dubiouscandle.dubiousdl;

public interface LossFunction {
	LossFunction SOFT_MAX_CEL = new LossFunction() {

		@Override
		public void getError(float[] output, float[] target, float[] out) {
			int numClasses = output.length;

			float[] softmaxOutput = new float[numClasses];
			float sumExp = 0.0f;
			for (int i = 0; i < numClasses; i++) {
				sumExp += Math.exp(output[i]);
			}
			for (int i = 0; i < numClasses; i++) {
				softmaxOutput[i] = (float) Math.exp(output[i]) / sumExp;
			}

			for (int i = 0; i < numClasses; i++) {
				out[i] = softmaxOutput[i] - target[i];
			}
		}

		@Override
		public float getLoss(float[] output, float[] target) {
			int numClasses = output.length;
			float loss = 0.0f;

			float sumExp = 0.0f;
			for (int i = 0; i < numClasses; i++) {
				sumExp += Math.exp(output[i]);
			}

			float[] softmaxOutput = new float[numClasses];
			for (int i = 0; i < numClasses; i++) {
				softmaxOutput[i] = (float) Math.exp(output[i]) / sumExp;
			}
			for (int i = 0; i < numClasses; i++) {
				if (target[i] == 1) {
					loss -= Math.log(softmaxOutput[i]);
				}
			}
			return loss;
		}
	};

	LossFunction BINARY_CEL = new LossFunction() {

		private static final float EPSILON = 1e-7f;

		@Override
		public void getError(float[] output, float[] target, float[] out) {
			float z = output[0];
			float y = target[0];

			float yHat = (float) (1.0 / (1.0 + Math.exp(-z)));
			out[0] = yHat - y;
		}

		@Override
		public float getLoss(float[] output, float[] target) {
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

	void getError(float[] output, float[] target, float[] out);

	float getLoss(float[] output, float[] target);
}

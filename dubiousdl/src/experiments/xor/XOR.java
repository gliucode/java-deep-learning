package experiments.xor;

import java.util.Random;

import com.dubiouscandle.dubiousnet.ActivationFunction;
import com.dubiouscandle.dubiousnet.AdamOptimizer;
import com.dubiouscandle.dubiousnet.Initializer;
import com.dubiouscandle.dubiousnet.LossFunction;
import com.dubiouscandle.dubiousnet.Matrix;
import com.dubiouscandle.dubiousnet.Model;

public class XOR {
	public static void main(String[] args) {
		long seed = System.nanoTime();
//		long seed = 43433447776916L;
		Random random = new Random(seed);

		Model model = new Model(
				new ActivationFunction[] { 
						ActivationFunction.LEAKY_RELU, 
						ActivationFunction.LEAKY_RELU,
						ActivationFunction.LEAKY_RELU, 
						ActivationFunction.IDENTITY,
				},
				new int[] { 2, 3, 3, 3, 1 }, 4, new Initializer.XaviarNormal(new Random(random.nextLong())));

		AdamOptimizer op = new AdamOptimizer(model, LossFunction.BINARY_CEL, .9f, .99f);

		for (int i = 0; i <= 300; i++) {
			Matrix inputs = new Matrix(2, 4);
			Matrix targets = new Matrix(1, 4);

			inputs.set(0, 0, 0);
			inputs.set(1, 0, 0);
			inputs.set(0, 1, 1);
			inputs.set(1, 1, 0);
			inputs.set(0, 2, 0);
			inputs.set(1, 2, 1);
			inputs.set(0, 3, 1);
			inputs.set(1, 3, 1);

			targets.set(0, 0, 0);
			targets.set(0, 1, 1);
			targets.set(0, 2, 1);
			targets.set(0, 3, 0);

			op.step(inputs, targets, 0.01f);

			if (i % 10 == 0) {
				Matrix outputs = new Matrix(1, 4);
				model.forwardPropagate(inputs, outputs);
				System.out.println("EPOCH: " + i);
				System.out.println("0 ^ 0 = " + sigmoid(outputs.get(0, 0)));
				System.out.println("1 ^ 0 = " + sigmoid(outputs.get(0, 1)));
				System.out.println("0 ^ 1 = " + sigmoid(outputs.get(0, 2)));
				System.out.println("1 ^ 1 = " + sigmoid(outputs.get(0, 3)));
				float[] output = { outputs.get(0, 0), outputs.get(0, 1), outputs.get(0, 2), outputs.get(0, 3) };
				float[] target = { 0, 1, 1, 0 };
				System.out.println("LOSS: " + LossFunction.BINARY_CEL.getLoss(output, target));
				System.out.println();
			}
		}

		System.out.println("SEED: " + seed);
	}

	private static float sigmoid(float x) {
		return (float) (1.0 / (1.0 + Math.exp(-x)));
	}
}

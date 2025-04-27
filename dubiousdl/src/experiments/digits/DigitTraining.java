package experiments.digits;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import com.dubiouscandle.dubiousnet.ActivationFunction;
import com.dubiouscandle.dubiousnet.AdamOptimizer;
import com.dubiouscandle.dubiousnet.Matrix;
import com.dubiouscandle.dubiousnet.Model;

public class DigitTraining {
	public static void main(String[] args) throws IOException {
		long seed = System.nanoTime();
		Random random = new Random(seed);

		ActivationFunction[] activations = { ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU,
				ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU, };
		int[] layerSizes = { 784, 256, 128, 64, 32, 10 };
		final int batchSize = 64;
		Digits digits = new Digits(MNIST.TRAINING_DIGITS, batchSize, random);
		Model model = new Model(activations, layerSizes, batchSize);
		AdamOptimizer op = new AdamOptimizer(model, 0.9f, .999f, true);
		Matrix output = new Matrix(10, batchSize);

		final int MAX_STEPS = 60_000 * 3 / 64;
		final int PRINT_RATE = 50;

		for (int step = 0; step < MAX_STEPS; step++) {
			digits.next();

			op.step(digits.input(), digits.target(), .001f);

			if (step % PRINT_RATE == 0) {
				model.forwardPropagate(digits.input(), output);
				float loss = computeCrossEntropyFromLogits(output, digits.target());
				System.out.printf("(%f,%.6f),\n", (float) step / MAX_STEPS, loss);

				if (loss < 0.05f) {
					break;
				}
			}
		}

		System.out.println("Final outputs after training:");
		model.forwardPropagate(digits.input(), output);
		for (int i = 0; i < batchSize; i++) {
			float[] digitImage = new float[28 * 28];
			for (int j = 0; j < 28 * 28; j++) {
				digitImage[j] = digits.input().get(j, i);
			}

			MNIST.printDigit(digitImage);

			float[] o = new float[10];
			for (int j = 0; j < 10; j++) {
				o[j] = output.get(j, i);
			}

			ActivationFunction.softmax(o);
			for (int j = 0; j < 10; j++) {
				System.out.println(j + " " + o[j]);
			}
		}

		try (FileOutputStream fileOut = new FileOutputStream("src/main/java/experiments/digitrecognizer.ser");
				ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
			out.writeObject(model);
			System.out.println("serialized");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private static float computeCrossEntropyFromLogits(Matrix logits, Matrix targets) {
		int batchSize = logits.cols;
		int numClasses = logits.rows;
		float loss = 0f;

		for (int i = 0; i < batchSize; i++) {
			float maxLogit = Float.NEGATIVE_INFINITY;
			for (int j = 0; j < numClasses; j++) {
				maxLogit = Math.max(maxLogit, logits.get(j, i));
			}

			float sumExp = 0f;
			for (int j = 0; j < numClasses; j++) {
				sumExp += Math.exp(logits.get(j, i) - maxLogit);
			}

			for (int j = 0; j < numClasses; j++) {
				if (targets.get(j, i) == 1f) {
					float logProb = logits.get(j, i) - maxLogit - (float) Math.log(sumExp);
					loss -= logProb;
				}
			}
		}

		return loss / batchSize;
	}
}

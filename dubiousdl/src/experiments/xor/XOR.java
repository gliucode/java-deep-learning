package experiments.xor;

import java.util.Random;

import com.dubiouscandle.dubiousdl.ActivationFunction;
import com.dubiouscandle.dubiousdl.Adam;
import com.dubiouscandle.dubiousdl.Batch;
import com.dubiouscandle.dubiousdl.Initializer;
import com.dubiouscandle.dubiousdl.LossFunction;
import com.dubiouscandle.dubiousdl.Matrix;
import com.dubiouscandle.dubiousdl.Model;

public class XOR {
	public static void main(String[] args) {
		long seed = System.nanoTime();
//		long seed = 43433447776916L;
		Random random = new Random(seed);

		Model model = new Model(
				new ActivationFunction[] { ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU,
						ActivationFunction.LEAKY_RELU, ActivationFunction.IDENTITY, },
				new int[] { 2, 3, 3, 3, 2 }, 4, new Initializer.XavierNormal(new Random(random.nextLong())));

		Batch batch = new Batch(new float[][] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, }, new int[] { 0, 1, 1, 0, }, 4,
				2, 2, 4, new Random(3));
		Adam op = new Adam(model, LossFunction.SOFT_MAX_CROSS_ENTROPY_LOSS, .9f, .99f);

		for (int i = 0; i <= 300_000; i++) {
		    batch.next();
		    Matrix input  = batch.input();
		    Matrix target = batch.target();

		    op.step(input, target, 0.001f);

		    if (i % 10_000 == 0) {
		        Matrix outputs = new Matrix(2, 4);
		        model.forwardPropagate(input, outputs);

		        System.out.println("EPOCH: " + i);

		        for (int col = 0; col < 4; col++) {
		            float p0 = outputs.get(0, col), p1 = outputs.get(1, col);
		            int pred = (p1 > p0 ? 1 : 0);
		            System.out.printf("%d ^ %d = %d   (p0=%.3f, p1=%.3f)%n",
		                (int)input.get(0,col),
		                (int)input.get(1,col),
		                pred, p0, p1);
		        }

		        float loss = LossFunction.SOFT_MAX_CROSS_ENTROPY_LOSS.getLoss(outputs, target);
		        System.out.println("LOSS: " + loss);
		        System.out.println();
		    }
		}

		System.out.println("SEED: " + seed);

		System.out.println("SEED: " + seed);
	}
}

package experiments.digits;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import com.dubiouscandle.dubiousnet.Matrix;

public class Digits {
	private final Random random;

	private Matrix input;
	private Matrix target;
	private ArrayList<Digit> digitList = new ArrayList<>();
	private int curIndex = 0;
	private final int batchSize;

	public Digits(float[][][] digits, int batchSize, Random random) {
		this.batchSize = batchSize;
		this.random = random;
		input = new Matrix(28 * 28, batchSize);
		target = new Matrix(10, batchSize);

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < digits[i].length; j++) {
				float[] digitImage = digits[i][j];
				int digit = i;
				digitList.add(new Digit(digitImage, digit));
			}
		}

		shuffle();
	}

	private void shuffle() {
		Collections.shuffle(digitList);

		for (int i = 0; i < digitList.size(); i++) {
			transform(digitList.get(i).digitImage);
		}
	}

	private void transform(float[] digitImage) {
//		float angle = 0;
		float angle = random.nextFloat(-(float) Math.PI / 6, (float) Math.PI / 6);

		float tx = -5 + random.nextFloat() * 10;
		float ty = -5 + random.nextFloat() * 10;

		float[] transformedImage = new float[28 * 28];
		float cosTheta = (float) Math.cos(angle);
		float sinTheta = (float) Math.sin(angle);

		for (int y = 0; y < 28; y++) {
			for (int x = 0; x < 28; x++) {
				float newX = cosTheta * (x - 14) - sinTheta * (y - 14) + 14 + tx;
				float newY = sinTheta * (x - 14) + cosTheta * (y - 14) + 14 + ty;

				newX = Math.max(0, Math.min(27, newX));
				newY = Math.max(0, Math.min(27, newY));

				int x1 = (int) newX;
				int y1 = (int) newY;
				int x2 = Math.min(27, x1 + 1);
				int y2 = Math.min(27, y1 + 1);

				float dx = newX - x1;
				float dy = newY - y1;

				float topLeft = digitImage[y1 * 28 + x1];
				float topRight = digitImage[y1 * 28 + x2];
				float bottomLeft = digitImage[y2 * 28 + x1];
				float bottomRight = digitImage[y2 * 28 + x2];

				float interpolatedValue = (1 - dx) * (1 - dy) * topLeft + dx * (1 - dy) * topRight
						+ (1 - dx) * dy * bottomLeft + dx * dy * bottomRight;

				transformedImage[y * 28 + x] = interpolatedValue;
			}
		}

		System.arraycopy(transformedImage, 0, digitImage, 0, 28 * 28);
	}

	public void next() {
		curIndex += batchSize;

		if (curIndex + batchSize > digitList.size()) {
			shuffle();
			curIndex = 0;
		}

		for (int i = 0; i < batchSize; i++) {
			Digit digit = digitList.get(curIndex + i);

			for (int j = 0; j < 28 * 28; j++) {
				input.set(j, i, digit.digitImage[j]);
			}
			for (int j = 0; j < 10; j++) {
				target.set(j, i, digit.digit == j ? 1 : 0);
			}
		}
	}

	public Matrix input() {
		return input;
	}

	public Matrix target() {
		return target;
	}

	private class Digit {
		Digit(float[] digitImage, int digit) {
			this.digitImage = digitImage;
			this.digit = digit;
		}

		float[] digitImage;
		private int digit;
	}
}

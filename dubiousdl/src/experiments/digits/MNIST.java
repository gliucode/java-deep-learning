package experiments.digits;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class MNIST {
	private static final Random RANDOM = new Random();
	public static final float[][][] TRAINING_DIGITS = new float[10][][]; // [digit][index][bits]
	public static final float[][][] TESTING_DIGITS = new float[10][][]; // [digit][index][bits]

	public static float[] getDigit(int digit) {
		int index = RANDOM.nextInt(TRAINING_DIGITS[digit].length);

		float[] originalImage = TRAINING_DIGITS[digit][index];

		int minX = Integer.MAX_VALUE;
		int minY = Integer.MAX_VALUE;
		int maxX = Integer.MIN_VALUE;
		int maxY = Integer.MIN_VALUE;

		for (int x = 0; x < 28; x++) {
			for (int y = 0; y < 28; y++) {
				if (originalImage[x * 28 + y] != 0) {
					minX = Math.min(x, minX);
					minY = Math.min(y, minY);
					maxX = Math.max(x, maxX);
					maxY = Math.max(y, maxY);
				}
			}
		}
		int tx = RANDOM.nextInt(-minX, 29 - maxX);
		int ty = RANDOM.nextInt(-minY, 29 - maxY);
		
		float[] digitImage = new float[28 * 28];
		for (int x = minX; x <= maxX; x++) {
		    for (int y = minY; y <= maxY; y++) {
		        int newX = x + tx;
		        int newY = y + ty;

		        if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
		        	digitImage[newX * 28 + newY] = originalImage[x * 28 + y];
		        }
		    }
		}

		return digitImage;
	}

	private static void loadDigits(float[][][] digits, File file) throws IOException {
		if (digits.length != 10) {
			throw new IllegalArgumentException("Invalid digits length.");
		}

		DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));

		for (int i = 0; i < 10; i++) {
			int n = in.readInt();
			digits[i] = new float[n][28 * 28];
		}

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < digits[i].length; j++) {
				for (int k = 0; k < 28 * 28; k++) {
					digits[i][j][k] = in.read() / 255f;
				}
			}
		}

		in.close();
	}

	public static void printDigit(float[] digit) {
		String gradient = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\\\|()1{}[]?-_+~<>i!lI;:,\\\"^`'.";

		gradient = new StringBuilder(gradient).reverse().toString();

		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				int index = i * 28 + j;
				char c = gradient.charAt((int) (digit[index] * gradient.length() - .001f));
				System.out.print(c);
				System.out.print(c);
			}
			System.out.println();
		}
	}

	public static void init() {
		try {
			loadDigits(TRAINING_DIGITS, new File("src/main/resources/mnist/mnist_training.bin"));
			loadDigits(TESTING_DIGITS, new File("src/main/resources/mnist/mnist_testing.bin"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	static {
		init();
	}

	protected static void parseDigitsToBin(File from, File to) throws IOException {
		Scanner in = new Scanner(from);
		DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(to)));

		@SuppressWarnings("unchecked")
		ArrayList<Integer>[] ints = new ArrayList[10];

		for (int i = 0; i <= 9; i++) {
			ints[i] = new ArrayList<>();
		}

		int counter = 0;
		while (in.hasNext()) {
			int digit = in.nextInt();

			for (int i = 0; i < 28 * 28; i++) {
				ints[digit].add(in.nextInt());
			}

			counter++;

			if (counter % 1000 == 0) {
				System.out.println(counter);
			}
		}
		in.close();
		for (int i = 0; i < 10; i++) {
			out.writeInt(ints[i].size() / (28 * 28));
		}

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < ints[i].size(); j++) {
				int x = (int) ints[i].get(j);

				out.write(x);
			}
		}
		out.flush();
		out.close();
	}
}

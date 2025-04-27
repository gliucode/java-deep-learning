package com.dubiouscandle.dubiousdl;

import java.util.Random;

public abstract interface Initializer {
	float get(int fan_in, int fan_out);

	public static class XaviarUniform implements Initializer {
		private Random random;

		public XaviarUniform(Random random) {
			this.random = random;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			float x = (float) Math.sqrt(6.0 / (fan_in + fan_out));
			return random.nextFloat(-x, x);
		}
	}

	public static class XaviarNormal implements Initializer {
		private Random random;

		public XaviarNormal(Random random) {
			this.random = random;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			float x = (float) Math.sqrt(2.0 / (fan_in + fan_out));
			return (float) random.nextGaussian(0, x);
		}
	}

	public static class HeNormal implements Initializer {
		private Random random;

		public HeNormal(Random random) {
			this.random = random;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			float x = (float) Math.sqrt(2.0 / fan_in);
			return (float) random.nextGaussian(0, x);
		}
	}

	public static class LeCunNormal implements Initializer {
		private Random random;

		public LeCunNormal(Random random) {
			this.random = random;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			float x = (float) Math.sqrt(1.0 / fan_in);
			return (float) random.nextGaussian(0, x);
		}
	}

	public static class LeCunUniform implements Initializer {
		private Random random;

		public LeCunUniform(Random random) {
			this.random = random;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			float x = (float) Math.sqrt(3.0 / fan_in);
			return random.nextFloat(-x, x);
		}
	}

	public static class Constant implements Initializer {
		private float value;

		public Constant(float value) {
			this.value = value;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			return value;
		}
	}

	public static class Uniform implements Initializer {
		private Random random;

		public Uniform(Random random) {
			this.random = random;
		}

		@Override
		public float get(int fan_in, int fan_out) {
			return random.nextFloat(-1, 1);
		}
	}
}

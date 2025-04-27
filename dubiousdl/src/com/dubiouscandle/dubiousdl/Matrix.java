package com.dubiouscandle.dubiousdl;

import java.io.Serializable;

public class Matrix implements Serializable {
	private static final long serialVersionUID = 3579437216201658179L;
	private final float[] data;
	private final int rows, cols;
	private boolean transposed = false;

	/**
	 * returns the underlying data array of this matrix. mutations to the returned array will be reflected in this object
	 * 
	 * @return the underlying data array of this matrix
	 */
	public float[] data() {
		return data;
	}

	/**
	 * creates a matrix with rows rows and cols columns.
	 * 
	 * @param rows
	 * @param cols
	 */
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		data = new float[rows * cols];
	}

	/**
	 * creates a copy of the inputed matrix
	 * 
	 * @param matrix
	 */
	public Matrix(Matrix matrix) {
		rows = matrix.rows;
		cols = matrix.cols;
		data = matrix.data.clone();
		transposed = matrix.transposed;
	}

	/**
	 * transposes this matrix by setting a flag.
	 */
	public void transpose() {
		transposed = !transposed;
	}

	/**
	 * computes the index inside the data array that row and col correspond to
	 * 
	 * @param row
	 * @param col
	 * @return the index in the data array
	 */
	private int getIndex(int row, int col) {
		if (transposed) {
			return col * cols + row;
		} else {
			return row * cols + col;
		}
	}

	/**
	 * gets the value at row row and column col
	 * 
	 * @param row
	 * @param col
	 * @return the value at row row and column col
	 */
	public float get(int row, int col) {
		checkIndex(row, col);

		return data[getIndex(row, col)];
	}

	/**
	 * sets the value at row row and column col to value
	 * 
	 * @param row
	 * @param col
	 * @param value
	 */
	public void set(int row, int col, float value) {
		checkIndex(row, col);

		data[getIndex(row, col)] = value;
	}

	/**
	 * throws an IndexOutOfBoundsException if row and col are invalid for this matrix
	 * 
	 * @param row
	 * @param col
	 */
	private void checkIndex(int row, int col) {
		if (row < 0 || row >= rows() || col < 0 || col >= cols()) {
			throw new IndexOutOfBoundsException(
					"Index (" + row + ", " + col + ") out of bounds for size (" + rows() + ", " + cols() + ").");
		}
	}

	/**
	 * @return the number of rows in this matrix
	 */
	public int rows() {
		return transposed ? cols : rows;
	}

	/**
	 * @return the number of cols in this matrix
	 */
	public int cols() {
		return transposed ? rows : cols;
	}

	/**
	 * computes the matrix multiplication of a and b and stores it in result
	 * 
	 * @param a
	 * @param b
	 * @param result
	 */
	public static void multiply(Matrix a, Matrix b, Matrix result) {
		if (a.cols() != b.rows()) {
			throw new IllegalArgumentException("Matrix dimensions do not match for dot product.");
		}
		if (result.rows() != a.rows() || result.cols() != b.cols()) {
			throw new IllegalArgumentException("Result matrix has incorrect dimensions.");
		}

		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < b.cols(); j++) {
				float sum = 0;
				for (int k = 0; k < a.cols(); k++) {
					sum += a.get(i, k) * b.get(k, j);
				}
				result.set(i, j, sum);
			}
		}
	}

	/**
	 * clears all values in this matrix to be 0
	 */
	public void clear() {
		for (int i = 0; i < data.length; i++) {
			data[i] = 0;
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(data.length * 4);

		for (int i = 0; i < rows(); i++) {
			sb.append('[');
			for (int j = 0; j < cols(); j++) {
				sb.append(get(i, j));
				if (j != cols() - 1) {
					sb.append(',').append(' ');
				}
			}
			sb.append(']').append('\n');
		}

		return sb.toString();
	}

	/**
	 * throws an IllegalArgumentException if matrix is not the same dimension as this one
	 * 
	 * @param matrix
	 */
	private void assertSameDimension(Matrix matrix) {
		if (rows() != matrix.rows() || cols() != matrix.cols()) {
			throw new IllegalArgumentException("Dimensions do not match.");
		}
	}

	/**
	 * sets all values in this matrix into the inputed matrix
	 * 
	 * @param matrix
	 * @throws IllegalArgumentException if the inputed matrix does not have the same dimension as this one
	 */
	public void set(Matrix matrix) {
		assertSameDimension(matrix);

		System.arraycopy(matrix.data, 0, data, 0, data.length);
	}

	/**
	 * multiplies all elements in this matrix by multiplier
	 * 
	 * @param multiplier
	 */
	public void multiply(float multiplier) {
		for (int i = 0; i < data.length; i++) {
			data[i] *= multiplier;
		}
	}

	/**
	 * adds addend to all elements in this matrix
	 * 
	 * @param addend
	 */
	public void add(float addend) {
		for (int i = 0; i < data.length; i++) {
			data[i] += addend;
		}
	}

	/**
	 * adds broadcasted the inputed matrix across this matrix.<br>
	 * the inputed matrix should atleast one dimension be of size 1 and have the non 1 dimension match the corresponding dimension in this
	 * 
	 * @param broadcast
	 */
	public void addBroadcasted(Matrix broadcast) {
		if (broadcast.rows() == 1 && broadcast.cols() == cols()) {
			for (int i = 0; i < rows(); i++) {
				for (int j = 0; j < cols(); j++) {
					set(i, j, get(i, j) + broadcast.get(0, j));
				}
			}
		} else if (broadcast.cols() == 1 && broadcast.rows() == rows()) {
			for (int i = 0; i < rows(); i++) {
				for (int j = 0; j < cols(); j++) {
					set(i, j, get(i, j) + broadcast.get(i, 0));
				}
			}
		} else if (broadcast.rows() == 1 && broadcast.cols() == 1) {
			float addend = broadcast.get(0, 0);
			add(addend);
		} else {
			throw new IllegalArgumentException("Invalid broadcast matrix dimensions.");
		}
	}

	/**
	 * clones the array and all matrices in the array. if any elements were null then it will keep them as null.
	 * 
	 * @param matrixArray
	 * @return a new Matrix[]
	 */
	public static Matrix[] deepClone(Matrix[] matrixArray) {
		Matrix cloned[] = matrixArray.clone();
		for (int i = 0; i < cloned.length; i++) {
			cloned[i] = cloned[i] == null ? null : new Matrix(cloned[i]);
		}
		return cloned;
	}

	/**
	 * multiplies all elements in a by the corresponding element in b
	 * 
	 * @param a
	 * @param b
	 */
	public static void hadamardProduct(Matrix a, Matrix b) {
		if (a.rows() != b.rows() || a.cols() != b.cols()) {
			throw new IllegalArgumentException("Dimension mismatch.");
		}
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				a.set(i, j, a.get(i, j) * b.get(i, j));
			}
		}
	}

	/**
	 * creates a copy of matrix of dimensions newRows x newCols. it will pad with 0 or truncate depending on the dimension.
	 * 
	 * @param matrix
	 * @param newRows
	 * @param newCols
	 * @return a copy of matrix
	 */
	public static Matrix copyOf(Matrix matrix, int newRows, int newCols) {
		Matrix copy = new Matrix(newRows, newCols);

		for (int i = 0; i < newRows; i++) {
			for (int j = 0; j < newCols; j++) {
				copy.set(i, j, matrix.get(i, j));
			}
		}

		return copy;
	}

	/**
	 * 
	 * @param matrix
	 * @return a new matrix of the same dimension as the inputed one
	 */
	public static Matrix emptyCopyOf(Matrix matrix) {
		return new Matrix(matrix.rows(), matrix.cols());
	}

	/**
	 * computs the sum of columns for this matrix and stores it in out
	 * 
	 * @param out
	 */
	public void sumOfCols(Matrix out) {
		if (out.cols() != 1 || out.rows() != rows()) {
			throw new IllegalArgumentException("Invalid out matrix size.");
		}
		for (int i = 0; i < out.data.length; i++) {
			out.data[i] = 0;
		}
		for (int i = 0; i < rows(); i++) {
			float s = 0;
			for (int j = 0; j < cols(); j++) {
				s += get(i, j);
			}
			out.set(i, 0, s);
		}
	}
}

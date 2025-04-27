package experiments.digits;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.TextArea;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.Rectangle2D;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.dubiouscandle.dubiousnet.ActivationFunction;
import com.dubiouscandle.dubiousnet.Matrix;
import com.dubiouscandle.dubiousnet.Model;

public class DigitRecognition {
	static {
		MNIST.init();
	}

	static Model model;
	static DrawPanel drawPanel = new DrawPanel();
	static TextArea textArea = new TextArea();

	public static void main(String[] args) {
		JFrame frame = new JFrame();
		textArea.setEditable(true);
		textArea.setEnabled(true);
		textArea.setPreferredSize(new Dimension(400, 100));
		textArea.setBackground(Color.WHITE);

		frame.add(textArea, BorderLayout.EAST);
		frame.add(drawPanel, BorderLayout.WEST);
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);

		try (FileInputStream fileIn = new FileInputStream("src/main/java/experiments/digitrecognizer.ser");
				ObjectInputStream in = new ObjectInputStream(fileIn)) {
			model = (Model) in.readObject();
			model = new Model(model, 1);
		} catch (Exception e) {
			e.printStackTrace();
		}

		drawPanel.addKeyListener(new KeyAdapter() {
			private Random random = new Random();

			public void keyPressed(KeyEvent e) {
				if (e.getKeyCode() == KeyEvent.VK_R) {
					for (int i = 0; i < drawPanel.pixels.length; i++) {
						drawPanel.pixels[i] = 0;
					}
					drawPanel.repaint();
				} else if (e.getKeyCode() == KeyEvent.VK_E) {
					int digit = random.nextInt(10);
					drawPanel.pixels = MNIST.getDigit(digit).clone();
					updateRankings();
					drawPanel.repaint();
				}
			}
		});

		drawPanel.setFocusable(true);
		drawPanel.requestFocusInWindow();
	}

	public static void updateRankings() {
		Matrix tmp1 = new Matrix(28 * 28, 1);
		Matrix tmp2 = new Matrix(10, 1);
		System.arraycopy(drawPanel.pixels, 0, tmp1.data, 0, 28 * 28);

		model.forwardPropagate(tmp1, tmp2);
		float[] out = Arrays.copyOf(tmp2.data, 10);

		ActivationFunction.softmax(out);

		@SuppressWarnings("unchecked")
		Entry<Float, Integer>[] entries = new Entry[10];
		for (int i = 0; i < out.length; i++) {
			float term = out[i];
			entries[i] = Map.entry(term, i);
		}
		Arrays.sort(entries, (a, b) -> -Float.compare(a.getKey(), b.getKey()));

		StringBuilder text = new StringBuilder();
		for (Entry<Float, Integer> entry : entries) {
			text.append(entry.getValue()).append(' ').append(entry.getKey()).append('\n');
		}

		textArea.setText(text.toString());
	}

	static float px, py;

	private static class DrawPanel extends JPanel {
		private float[] pixels = new float[28 * 28];

		DrawPanel() {
			setBackground(Color.gray);

			setPreferredSize(new Dimension(400, 400));

			addMouseMotionListener(new MouseMotionAdapter() {
				@Override
				public void mouseDragged(MouseEvent e) {
					if (e.getX() >= getWidth() || e.getX() < 0 || e.getY() >= getWidth() || e.getY() < 0) {
						return;
					}
					float y = 28 * (float) e.getX() / getWidth();
					float x = 28 * (float) e.getY() / getWidth();

					float dist;

					do {
						dist = (float) Math.hypot(x - px, y - py);

						draw(px, py);

						float nx = (x - px) / dist;
						float ny = (y - py) / dist;

						px += nx * .3f;
						py += ny * .3f;
					} while (dist > .3f);
					draw(x, y);
					updateRankings();
					repaint();

					px = x;
					py = y;
				}
			});
			addMouseListener(new MouseAdapter() {
				@Override
				public void mousePressed(MouseEvent e) {
					drawPanel.requestFocusInWindow();
					if (e.getX() >= getWidth() || e.getX() < 0 || e.getY() >= getWidth() || e.getY() < 0) {
						return;
					}

					float y = 28 * (float) e.getX() / getWidth();
					float x = 28 * (float) e.getY() / getWidth();
					draw(x, y);
					updateRankings();
					repaint();
					px = x;
					py = y;
				}
			});

		}

		private void draw(float x, float y) {
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					float dist = (float) Math.hypot(x - i, y - j);
					float deltaThickness = 1.0f - dist / 1.5f;
					if (dist > 1.5f) {
						deltaThickness = 0;
					}

					deltaThickness *= 0.3f;

					pixels[28 * i + j] = Math.min(pixels[28 * i + j] + deltaThickness, 1);
				}
			}
		}

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);

			float w = getWidth() / 28f;

			Graphics2D g2d = (Graphics2D) g;

			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					float brightness = pixels[i * 28 + j];
					g2d.setColor(new Color(brightness, brightness, brightness));

					float x = j * w;
					float y = i * w;
					Shape rect = new Rectangle2D.Float(x, y, w, w);
					g2d.fill(rect);
				}
			}
		}

		private static final long serialVersionUID = 1513754492484967363L;

	}
}

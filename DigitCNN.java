import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import javax.imageio.ImageIO;

/**
 * a convolutional neural network for classifying handwritten digits
 */
public class DigitCNN {
   /** the number of epochs to train on the training data */
   public static final int EPOCH_COUNT = 1;

   /** the number of channels in the image data */
   public static final int CHANNEL_COUNT = 1;

   /** the size of the image (assumed to be square) */
   public static final int IMAGE_SIZE = 28;

   /** the number of output neurons */
   public static final int OUTPUT_SIZE = 10;

   /** the number of filters used in the convolutional layer */
   public static final int FILTER_COUNT = 32;

   /** the size of each square convolutional filter */
   public static final int FILTER_SIZE = 3;

   /** the stride length used in convolutions */
   public static final int STRIDE = 2;

   /** the amount of padding applied to the input image during convolution */
   public static final int PADDING = 0;

   /** the number of hidden layers */
   public static final int HIDDEN_LAYER_COUNT = 1;

   /** the percent of input nodes used to calculate the number of hidden nodes */
   public static final double HIDDEN_FACTOR = 0.3;

   /** the rate at which weight updates are learned during training */
   public static final double LEARN_RATE = 0.1;

   /** the file path to the training images */
   public static final String TRAIN_IMAGES_PATH = "data/mnist_train.csv";

   /** the file path to the test images */
   public static final String TEST_IMAGES_PATH = "data/mnist_test.csv";

   /** the output directory path */
   public static final String OUTPUT_PATH = "output/";

   /** the input, hidden, and output layers */
   private ArrayList<double[]> layers;

   /** the weight matrices between the input, hidden, and output layers */
   private ArrayList<double[][]> weights;

   /** the filters used for convolutions */
   private double[][][] filters;

   /**
    * the default constructor for DigitCNN that intializes all layers, weights, and
    * filters
    */
   public DigitCNN() {
      layers = new ArrayList<>();
      weights = new ArrayList<>();
      int convOutputSize = (IMAGE_SIZE - FILTER_SIZE + 2 * PADDING) / STRIDE + 1;
      int inputSize = CHANNEL_COUNT * FILTER_COUNT * convOutputSize * convOutputSize;
      int hiddenSize = (int) (inputSize * HIDDEN_FACTOR);

      layers.add(new double[inputSize]);
      if (HIDDEN_LAYER_COUNT == 0) {
         weights.add(generateWeightMatrix(inputSize, OUTPUT_SIZE));

      } else if (HIDDEN_LAYER_COUNT > 0) {
         weights.add(generateWeightMatrix(inputSize, hiddenSize));
         layers.add(new double[hiddenSize]);
         for (int i = 0; i < HIDDEN_LAYER_COUNT - 1; i++) {
            weights.add(generateWeightMatrix(hiddenSize, hiddenSize));
            layers.add(new double[hiddenSize]);
         }
         weights.add(generateWeightMatrix(hiddenSize, OUTPUT_SIZE));
      }
      layers.add(new double[OUTPUT_SIZE]);

      filters = new double[FILTER_COUNT][FILTER_SIZE][FILTER_SIZE];
      Random random = new Random();
      for (int i = 0; i < filters.length; i++) {
         for (int j = 0; j < filters[0].length; j++) {
            for (int k = 0; k < filters[0][0].length; k++) {
               filters[i][j][k] = random.nextGaussian();
            }
         }
      }
   }

   /**
    * generates a randomly initialized weight matrix with given input and output
    * sizes
    * 
    * @param inputSize  the size of the input
    * @param outputSize the size of the output
    * @return the generated weight matrix
    */
   public static double[][] generateWeightMatrix(int inputSize, int outputSize) {
      Random random = new Random();

      double[][] layerWeights = new double[inputSize][outputSize];
      for (int i = 0; i < inputSize; i++) {
         for (int j = 0; j < outputSize; j++) {
            layerWeights[i][j] = random.nextGaussian();
         }
      }

      return layerWeights;
   }

   /**
    * parses image data and labels from a specified file
    * 
    * @param filePath the specified file path
    * @return the parsed images
    */
   public static ArrayList<Image> readImageData(String filePath) {
      ArrayList<Image> images = new ArrayList<>();

      try {
         BufferedReader reader = new BufferedReader(new FileReader(filePath));
         reader.readLine();

         String line = reader.readLine();
         while (line != null) {
            String[] values = line.split(",");

            double[][][] data = new double[CHANNEL_COUNT][IMAGE_SIZE][IMAGE_SIZE];
            for (int i = 0; i < CHANNEL_COUNT; i++) {
               for (int r = 0; r < IMAGE_SIZE; r++) {
                  for (int c = 0; c < IMAGE_SIZE; c++) {
                     data[i][r][c] = Integer.parseInt(values[r * IMAGE_SIZE + c]) / 255;
                  }
               }
            }

            Image image = new Image();
            image.label = Integer.parseInt(values[0]);
            image.data = data;
            images.add(image);

            line = reader.readLine();
         }

         reader.close();

      } catch (IOException e) {
         System.err.println(e);
         System.exit(1);
      }

      return images;
   }

   /**
    * calculates the softmax probabilities for a given vector input
    * 
    * @param input the input vector
    * @return the array of softmax probabilities
    */
   public static double[] softmax(double[] input) {
      double[] output = new double[input.length];
      double sum = 0;

      for (int i = 0; i < input.length; i++) {
         output[i] = Math.exp(input[i]);
         sum += output[i];
      }

      for (int i = 0; i < output.length; i++) {
         output[i] /= sum;
      }

      return output;
   }

   /**
    * calculates the derivative of the softmax function with respect to a given
    * vector input
    * 
    * @param input the input vector
    * @return the Jacobian matrix that is the derivative of the softmax function
    */
   public static double[][] softmaxDerivative(double[] input) {
      double[][] jacobian = new double[input.length][input.length];
      double[] softmaxValues = softmax(input);

      for (int r = 0; r < input.length; r++) {
         for (int c = 0; c < input.length; c++) {
            if (r == c) {
               jacobian[r][c] = softmaxValues[r] * (1 - softmaxValues[r]);
            } else {
               jacobian[r][c] = -softmaxValues[r] * softmaxValues[c];
            }
         }
      }

      return jacobian;
   }

   /**
    * calculates the ReLU function for a single input value
    * 
    * @param input the input value
    * @return the output of the ReLU function
    */
   public static double relu(double input) {
      return Math.max(0, input);
   }

   /**
    * calcuates the derivative of the ReLU function with respect to a single input
    * value
    * 
    * @param input the input value
    * @return the derivative of the ReLU function
    */
   public static double reluDerivative(double input) {
      double derivative = 0;

      if (input >= 0) {
         derivative = 1;
      }

      return derivative;
   }

   /**
    * calculates the categorical cross-entropy loss given predicted class
    * probabilities and the correct label
    * 
    * @param probabilities the array of predicted class probabilities
    * @param label         the correct label
    * @return the categorical cross-entropy loss
    */
   public static double calculateLoss(double[] probabilities, int label) {
      double[] labels = new double[probabilities.length];
      labels[label] = 1;

      double loss = 0;
      for (int i = 0; i < probabilities.length; i++) {
         loss += labels[i] * Math.log(probabilities[i] + Double.MIN_VALUE);
      }

      loss *= -1;

      return loss;
   }

   /**
    * calculates the derivative of categorical cross-entropy loss with respect to
    * the predicted class probabilities
    * 
    * @param probabilities the array of predicted class probabilities
    * @param label         the correct label
    * @return the derivative of the categorical cross-entropy loss
    */
   public static double[] calculateLossDerivative(double[] probabilities, int label) {
      double[] lossDerivative = new double[probabilities.length];

      for (int i = 0; i < probabilities.length; i++) {
         if (i == label) {
            lossDerivative[i] = probabilities[i] - 1;
         } else {
            lossDerivative[i] = probabilities[i];
         }
      }

      return lossDerivative;
   }

   /**
    * performs forward propagation
    * 
    * @param input a 3D array represingting the input image
    * @return an array of predicted probabilities for each class
    */
   public double[] forwardPropagate(double[][][] input) {
      int convOutputSize = (IMAGE_SIZE - FILTER_SIZE + 2 * PADDING) / STRIDE + 1;
      int inputSize = CHANNEL_COUNT * FILTER_COUNT * convOutputSize * convOutputSize;
      int hiddenSize = (int) (inputSize * HIDDEN_FACTOR);

      double[][][] convOutput = new double[FILTER_COUNT][convOutputSize][convOutputSize];
      for (int f = 0; f < FILTER_COUNT; f++) {
         for (int c = 0; c < CHANNEL_COUNT; c++) {
            for (int i = 0; i < convOutputSize; i++) {
               for (int j = 0; j < convOutputSize; j++) {
                  for (int k = 0; k < FILTER_SIZE; k++) {
                     for (int l = 0; l < FILTER_SIZE; l++) {
                        convOutput[f][i][j] += input[c][i * STRIDE + k][j * STRIDE + l] * filters[f][k][l];
                     }
                  }
               }
            }
         }
      }

      for (int i = 0; i < input.length; i++) {
         for (int j = 0; j < input[0].length; j++) {
            for (int k = 0; k < input[0][0].length; k++) {
               int idx = i * (input[0].length * input[0][0].length) + j * input[0][0].length + k;
               layers.get(0)[idx] = input[i][j][k];
            }
         }
      }

      for (int w = 0; w < weights.size(); w++) {
         double weightedSum = 0;
         double[][] weightMatrix = weights.get(w);
         for (int i = 0; i < weightMatrix[0].length; i++) {
            for (int j = 0; j < weightMatrix.length; j++) {
               weightedSum += layers.get(w)[j] * weightMatrix[j][i];
            }
            layers.get(w + 1)[i] = relu(weightedSum);
         }
      }

      double[] output = layers.get(weights.size());
      return softmax(output);
   }

   /**
    * performs backpropagation and updates weights
    * 
    * @param gradient the derivative of the loss function with respect to the
    *                 output of forward propagation
    */
   public void backPropagate(double[] gradient) {
      // :(
   }

   /**
    * trains the CNN on a set of training images for a specified number of epochs
    */
   public void train() {
      ArrayList<Image> images = readImageData(TRAIN_IMAGES_PATH);
      for (int e = 0; e < EPOCH_COUNT; e++) {
         int matches = 0;
         for (int i = 0; i < images.size(); i++) {
            Image image = images.get(i);
            double[] output = forwardPropagate(image.data);
            int maxProbIndex = 0;
            for (int o = 0; o < output.length; o++) {
               if (output[o] > output[maxProbIndex]) {
                  maxProbIndex = o;
               }
            }
            System.out.println("guess: " + maxProbIndex + ", label: " + image.label);
            if (maxProbIndex == image.label) {
               matches++;
            }
            double[] gradient = calculateLossDerivative(output, image.label);
            backPropagate(gradient);
         }
         System.out.println(100 * (double) matches / images.size());
      }
   }

   /**
    * tests the CNN on a set of test images and prints the prediction accuracy
    */
   public void test() {
      ArrayList<Image> images = readImageData(TEST_IMAGES_PATH);
      int matches = 0;
      for (int i = 0; i < images.size(); i++) {
         Image image = images.get(i);
         double[] output = forwardPropagate(image.data);
         int maxProbIndex = 0;
         for (int o = 0; o < output.length; o++) {
            if (output[o] > output[maxProbIndex]) {
               maxProbIndex = o;
            }
         }
         if (maxProbIndex == image.label) {
            matches++;
         }
      }
      System.out.println(100 * (double) matches / images.size());
   }

   public static void main(String[] args) {
      DigitCNN digitCNN = new DigitCNN();
      digitCNN.train();
      digitCNN.test();
   }

   /**
    * represents an image input for the CNN with an associated label
    */
   public static class Image {
      /** the correct classification of the image */
      public int label;

      /** the pixel data of the image */
      public double[][][] data;

      /**
       * writes the image data to a file
       * 
       * @param filename the name of the file to be saved
       */
      public void writeToFile(String filename) {
         try {
            BufferedImage image = new BufferedImage(data[0].length, data.length,
                  BufferedImage.TYPE_BYTE_GRAY);
            for (int i = 0; i < CHANNEL_COUNT; i++) {
               for (int r = 0; r < data.length; r++) {
                  for (int c = 0; c < data[0].length; c++) {
                     int color = (int) (data[i][r][c] * 255);
                     image.setRGB(c, r, new Color(color, color, color).getRGB());
                  }
               }
            }
            String[] parts = filename.split("\\.");
            String format = (parts.length == 2) ? parts[1] : "jpg";
            ImageIO.write(image, format, new File(OUTPUT_PATH + parts[0] +
                  "." + format));
         } catch (IOException e) {
            System.err.println(e);
            System.exit(1);
         }
      }
   }
}
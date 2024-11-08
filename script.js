// DesignDetective's JavaScript file.
// Import TensorFlow pretrained AI.
import * as tf from "@tensorflow/tfjs";
const fontEmbeddings = await tf.loadLayersModel(
  "https://example.com/fontembeddings.json",
);

// Function used to analyse a webpage:
async function analyzePage(url) {
  const html = await fetch(url).then((response) => response.text());
  const textData = html.replace(/\s+/g, "").toLowerCase();
  // feed the data to the TensorFlow AI:
  const tensor = tf.tensor2d([textData], [1, textData.length]);
  const fontEmbeddingsOutput = await fontEmbeddings.predict(tensor);
  // Extract the predicted font embeddings
  const predictions = fontEmbeddingsOutput.arraySync();
  // Log or save the results (e.g., to a local file)
  console.log(predictions);
}
// Call the analyzePage function with a URL
analyzePage("https://example.com");

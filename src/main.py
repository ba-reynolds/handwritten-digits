import numpy as np
from PIL import Image

import mnist_loader
import network


def main() -> None:
    weights_path = "weights.json"
    img_path = "assets/img.png"

    try:
        net = network.NetworkSerializer.load(weights_path)
    except FileNotFoundError:
        training_data = mnist_loader.load_images_and_labels("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
        test_data = mnist_loader.load_images_and_labels("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")

        net = network.Network([784, 30, 10], network.ReLU, network.CrossEntropyCost)
        net.SGD(
            training_data=training_data,
            epochs=25,
            mini_batch_size=10,
            eta=0.05,
            lmbda=5.0,
            test_data=test_data
        )
        network.NetworkSerializer.save(net, weights_path)

    # Number of items to show
    n_show = 3

    while True:
        # Wait for user input
        input(">>> ")

        # Open image, convert into black and white
        image = Image.open(img_path).convert("L")

        # Verify the dimensions are 28x28
        if image.size != (28, 28):
            print(f"Expected image dimensions (28, 28) got {image.size}")
            continue

        # Normalize values [0,255] --> [0, 1] and reshape the image so that
        # it fits with the number of neurons in the input layer
        image = (np.array(image) / 255).reshape(784, 1)

        # Pass image through the neural network
        result = net.feedforward(image)

        # Get 3 most likely results
        result = result.flatten()
        sorted_indices = np.argsort(result, axis=0)[::-1]
        top_indices = sorted_indices[:n_show]
        top_values = result[top_indices]

        print(f"Top predictions:")
        for i in range(n_show):
            print(f"    * Digit {top_indices[i]}, confidence: {top_values[i]:.2f}")

if __name__ == "__main__":
    main()


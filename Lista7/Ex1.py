import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, n_epochs=100):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        for _ in range(self.n_epochs):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)


def train_test(inputs, target, n_inputs, function_name):
    classifier = Perceptron(n_inputs)
    classifier.train(inputs, target)

    print(f"{function_name} com {n_inputs} entradas:")
    for i in range(len(inputs)):
        prediction = classifier.predict(inputs[i])
        print(f"Entrada: {inputs[i]}, Saida Esperada: {target[i]}, Saida do Perceptron: {prediction}")
    print("\n")


if __name__ == "__main__":
    # AND
    inputs_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_and = np.array([0, 0, 0, 1])
    train_test(inputs_and, target_and, 2, "AND")
   
    # OR
    inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_or = np.array([0, 1, 1, 1])
    train_test(inputs_or, target_or, 2, "OR")

    # XOR
    inputs_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_xor = np.array([0, 1, 1, 0])
    train_test(inputs_xor, target_xor, 2, "XOR")

    # OR - 10 entradas
    inputs_or_10 = np.random.randint(2, size=(1000, 10))
    target_or_10 = np.any(inputs_or_10, axis=1)
    train_test(inputs_or_10, target_or_10, 10, "OR (10 entradas)")

    # AND - 10 entradas
    inputs_and_10 = np.random.randint(2, size=(1000, 10))
    target_and_10 = np.all(inputs_and_10, axis=1)
    train_test(inputs_and_10, target_and_10, 10, "AND (10 entradas)")
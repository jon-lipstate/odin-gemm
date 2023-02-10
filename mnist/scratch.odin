package scratch
import "core:fmt"
import rnd "core:math/rand"
import "core:mem"
import "core:encoding/endian"

main :: proc() {
	// network := make_network({784, 16, 16, 10})
	// fmt.println(network.biases)
	// fmt.println(network.weights)
	training_set := read_images("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")
	test_set := read_images("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte")
	fmt.println(len(training_set), len(test_set))
}

relu :: #force_inline proc(value: $T) {
	return max(0, v)
}
MNIST_Image :: struct {
	data:     [784]u8, // 28x28 pixels
	expected: u8, // 0-9
}
Neural_Network :: struct {
	n_layers:          int,
	neurons_per_layer: []int,
	biases:            [][]f32,
	weights:           [][]f32,
}

make_network :: proc(neurons_per_layer: []int) -> Neural_Network {
	n_layers := len(neurons_per_layer)

	n := Neural_Network {
		n_layers          = n_layers,
		neurons_per_layer = neurons_per_layer,
		biases            = make_2d_jagged(neurons_per_layer, f32),
		weights           = make_2d_jagged(neurons_per_layer, f32),
	}
	init_rand(1234, n.biases)
	init_rand(5678, n.weights)
	return n
}

feed_forward :: proc(net: ^Neural_Network, activation: [][]f32) {
	assert(len(activation) == net.n_layers)
	for i := 0; i < net.n_layers; i += 1 {
		biases := net.biases[i]
		weights := net.weights[i]
		a := activation[i]
		assert(len(a) == len(biases))
		for j := 0; j < len(biases); j += 1 {
			a[j] = a[j] * weights[j] + biases[j]
		}
	}
}

// Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x.
// ``nabla_b`` and  ``nabla_w`` are layer-by-layer lists of numpy arrays, similar  to ``self.biases`` and ``self.weights``.
backprop :: proc(net: ^Neural_Network, x: $T, y: T) {
	nabla_b := make_2d_jagged(net.neurons_per_layer, f32)
	nabla_w := make_2d_jagged(net.neurons_per_layer, f32)
	defer delete(nabla_b) // TODO: does this get the backing too??
	defer delete(nabla_w) // TODO: RECYCLE this memory

	// feedforward
	previous_activation := x
	activations := make_2d_jagged(net.neurons_per_layer, f32)
	potentials := make_2d_jagged(net.neurons_per_layer, f32) //Zee's
	// set initial inputs as the activations?
	// start at first hidden layer?
	for i := 1; i < net.n_layers; i += 1 {
		row_len := net.neurons_per_layer[i]
		for j := 0; j < row_len; j += 1 {
			z := net.weights[i][j] * previous_activation[i][j] + net.biases[i][j]
			potentials[i][j] = z
		}
	}

	// activations = [x] # list to store all the activations, layer by layer
	// zs = [] # list to store all the z vectors, layer by layer
	// for b, w in zip(self.biases, self.weights):
	// 	z = np.dot(w, activation)+b
	// 	zs.append(z)
	// 	activation = sigmoid(z)
	// 	activations.append(activation)
	// # backward pass
	// delta = self.cost_derivative(activations[-1], y) * \
	// 	sigmoid_prime(zs[-1])
	// nabla_b[-1] = delta
	// nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	// # Note that the variable l in the loop below is used a little
	// # differently to the notation in Chapter 2 of the book.  Here,
	// # l = 1 means the last layer of neurons, l = 2 is the
	// # second-last layer, and so on.  It's a renumbering of the
	// # scheme in the book, used here to take advantage of the fact
	// # that Python can use negative indices in lists.
	// for l in xrange(2, self.num_layers):
	// 	z = zs[-l]
	// 	sp = sigmoid_prime(z)
	// 	delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	// 	nabla_b[-l] = delta
	// 	nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	// return (nabla_b, nabla_w)

}

//Train the neural Network using mini-batch stochastic gradient descent.  
// The "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired outputs.  
// The other non-optional parameters are  self-explanatory.  
// If "test_data" is provided then the network will be evaluated against the test data after each epoch, 
// eta: Learning Rate
stochastic_gradient_descent :: proc(
	net: ^Neural_Network,
	training_data: $T,
	epochs: int,
	mini_batch_size: int,
	eta: f32,
	test_data: ^T = nil,
) {
	n_test: int = 0
	if test_data != nil {
		n_test = len(test_data)
	}
	n_trainig_samples := len(training_data)
	n_mini_batches := n_trainig_samples / mini_batch_size

	r := rnd.create(1234)
	for e := 0; e < epochs; e += 1 {
		rnd.shuffle(training_data, &r)

	}
	// for j in xrange(epochs):
	// 	random.shuffle(training_data)
	// 	mini_batches = [
	// 		training_data[k:k+mini_batch_size]
	// 		for k in xrange(0, n, mini_batch_size)]
	// 	for mini_batch in mini_batches:
	// 		self.update_mini_batch(mini_batch, eta)
	// 	if test_data:
	// 		print "Epoch {0}: {1} / {2}".format(
	// 			j, self.evaluate(test_data), n_test)
	// 	else:
	// 		print "Epoch {0} complete".format(j)
}

update_mini_batch :: proc(net: ^Neural_Network, mini_batch: $T, eta: f32) {
	nabla_b := make_2d_jagged(net.neurons_per_layer, f32)
	nabla_w := make_2d_jagged(net.neurons_per_layer, f32)
	defer delete(nabla_b) // TODO: does this get the backing too??
	defer delete(nabla_w) // TODO: RECYCLE this memory
	delta_nabla_b, delta_nabla_w := backprop()

	scale := -1 * eta / len(mini_batch)
	hadd_jagged(nabla_b, delta_nabla_b, scale)
	hadd_jagged(nabla_w, delta_nabla_w, scale)
}
// v+=a*scale
hadd_jagged :: proc(value: [][]f32, addend: [][]f32, scale: f32 = 1.) {
	for i := 0; i < len(value); i += 1 {
		row := value[i]
		for j := 0; j < len(row); j += 1 {
			row[j] = scale * addend[i][j]
		}
	}
}
make_2d_jagged :: proc(len_per_row: []int, $T: typeid, allocator := context.allocator) -> [][]T {
	context.allocator = allocator
	n_layers := len(len_per_row)
	total_count := 0
	for i := 0; i < n_layers; i += 1 {
		total_count += len_per_row[i]
	}
	backing := make([]T, total_count)

	mem.zero_slice(backing) // make flag to zero or rand?

	res := make([][]T, n_layers)

	current_offset := 0
	for i in 0 ..< n_layers {
		res[i] = backing[current_offset:][:len_per_row[i]]
		current_offset += len_per_row[i]
	}
	return res
}
// np.random.randn function to generate Gaussian distributions with mean 0  and standard deviation  1
init_rand :: proc(seed: u64, mat: [][]$T) {
	r := rnd.create(seed)
	for i := 0; i < len(mat); i += 1 {
		row := mat[i]
		for j := 0; j < len(row); j += 1 {
			row[j] = rnd.float32(&r)
		}
	}
}

// Input x: Set the corresponding activation a1 for the input layer.
// Feedforward: For each l=2,3,... L
// zl= a(l-1) * wl + bl (Current Potential = Previous Activation * current weight + current bias)
// al=σ(zl)  (Current Activation = ReLU(Current Potential))

// Output error δL : Compute the vector δL=∇aC⊙σ′(zL)
// Cost Gradient hadamard sig prime??
// Backpropagate the error: For each l=L-1,L-2,...2
//  compute δl=((wl+1)Tδl+1)⊙σ′(zl)

// Output: The gradient of the cost function is given by ∂C∂wljk=al−1kδlj
//  and ∂C∂blj=δlj
// .

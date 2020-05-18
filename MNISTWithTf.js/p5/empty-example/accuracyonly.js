let mnist;
let inputs = [];
let train_image;

function setup() {
	createCanvas(200, 200);
	train_image = createImage(28, 28);
	loadMNIST(function (data) {
		mnist = data;
		console.log('data is ready.');
		// console.log(mnist);
	})
}

function draw() {
	let inputs = [];
	if (mnist) {
		//test data
		inputs_test = tf.tensor2d(mnist.test_images.slice(0, 10000));
		inputs_test = tf.div(inputs_test,tf.scalar(255.0));
		inputs_test = inputs_test.reshape([10000, 28, 28, 1]);
		outputs_test = tf.tensor1d(mnist.test_labels.slice(0, 10000));
		print(outputs_test.shape);

		async function test() {
			const model = await tf.loadLayersModel('indexeddb://my-model-4');

			console.log('Prediction from loaded model:');
			output_tem = model.predict(inputs_test);
			// output_tem.print();
			// tf.softmax(output_tem).print();
			label = tf.argMax(output_tem, 1);
			// tf.add(label, 1).print();
			tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();
		}
		test();
		noLoop();
	}
}
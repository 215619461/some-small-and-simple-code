let mnist;
let user_digit;
let user_has_drawing = false;
let user_guess_ele;
let model;
let label;

function mouseDragged() {
	user_has_drawing = true;
	user_digit.stroke(255);
	user_digit.strokeWeight(16);
	user_digit.line(mouseX, mouseY, pmouseX, pmouseY);
}

function keyPressed() {
	if (key == ' ') {
		user_has_drawing = false;
		user_digit.background(0);
	}
}

function setup() {
	createCanvas(200, 200).parent('container');
	user_digit = createGraphics(200, 200);

	user_guess_ele = select('#user_guess');

	loadMNIST(function (data) {
		mnist = data;
		console.log('data is ready.');
	})
}

function guessUserDigit() {
	let img = user_digit.get();
	if (!user_has_drawing) {
		user_guess_ele.html('_');
		return img;
	}
	let inputs = [];
	img.resize(28, 28);
	img.loadPixels();
	for (let i = 0; i < 784; i++) {
		inputs[i] = img.pixels[i * 4] / 255;
	}
	inputs = tf.tensor2d([inputs]);
	inputs = inputs.reshape([1, 28, 28, 1]);
	let prediction = model.predict(inputs);
	label = tf.argMax(prediction, 1);
	user_guess_ele.html(label.dataSync());
	console.log(label.dataSync()[0]);
	return img;
}

function draw() {
	background(0);
	if (mnist) {
		async function recognize() {
			model = await tf.loadLayersModel('indexeddb://my-model-3');

			guessUserDigit();
		}

		if (user_has_drawing) {
			image(user_digit, 0, 0);
			recognize();
		}
	}
}
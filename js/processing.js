var model;

async function loadModel(){
    model = await tf.loadLayersModel('../cnn/model.json');
//    console.log("Model Loaded")
}

async function predictImage(){

	 var imageData = canvas.toDataURL();

    // preprocess canvas
     const X = tf.browser.fromPixels(canvas)
            .resizeNearestNeighbor([28, 28])
            .mean(2)
            .expandDims(2)
            .expandDims()
            .toFloat();
     X.div(255.0);

//    console.log(`Shape of tensor: ${X.shape}`);
//    console.log(`dtype of Tensor: ${X.dtype}`);

    const predictions = await model.predict(X).data();
//    console.log(predictions);
    const arr = Array.from(predictions);
//    console.log(arr);

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
        }
    }
    console.log(maxIndex);

    return maxIndex;
}


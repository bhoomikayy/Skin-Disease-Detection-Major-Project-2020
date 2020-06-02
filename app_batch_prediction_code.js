


async function model_makePrediction(fname) {
	
	//console.log('met_cancer');
	
	
	let image = undefined;
	
	image = $('#selected-image').get(0);
	
	
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224])
	.toFloat();
	
	
	let offset = tf.scalar(127.5);
	
	tensor = tensor.sub(offset)
	.div(offset)
	.expandDims();

	
	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { 
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
		
	
	$("#prediction-list").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">
	${fname}</li>`);
	
	
	top5.forEach(function (p) {
	
		$("#prediction-list").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
	
		
	});
	
	
	$("#prediction-list").append(`<br>`);
		
}





function model_delay() {
	
	return new Promise(resolve => setTimeout(resolve, 200));
}


async function model_delayedLog(item, dataURL) {
	
	
	await model_delay();
	
	
	$("#selected-image").attr("src", dataURL);
	$("#displayed-image").attr("src", dataURL); //#########
	
	
	//console.log(item);
}



async function model_processArray(array) {
	
	for(var item of fileList) {
		
		
		let reader = new FileReader();
		
		
		let file = undefined;
	
		
		reader.onload = async function () {
			
			let dataURL = reader.result;
			
			await model_delayedLog(item, dataURL);
			
			
			
			var fname = file.name;
			
			
			$("#prediction-list").empty();
			
			
			await model_makePrediction(fname);
		}
		
		file = item;
		
			
		reader.readAsDataURL(file);
	}
}














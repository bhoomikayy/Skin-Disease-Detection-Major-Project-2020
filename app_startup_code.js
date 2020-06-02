

function simulateClick(tabID) {
	
	document.getElementById(tabID).click();
}



function predictOnLoad() {
	
	
	setTimeout(simulateClick.bind(null,'predict-button'), 500);
}








let model;
(async function () {
	
	model = await tf.loadModel('https://amazing-hawking-bc90ad.netlify.app//model.json');
	$("#selected-image").attr("src", "http://skin.test.woza.work/assets/samplepic.jpg");
	
	
	$('.progress-bar').hide();
	
	
	
	predictOnLoad();
	
	
	
})();



	
$("#predict-button").click(async function () {
	
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
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
	

		
		var file_name = 'samplepic.jpg';
		$("#prediction-list").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);
		
	
		top5.forEach(function (p) {
		
		
			$("#prediction-list").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
		
			
		});
	
	
});





$("#image-selector").change(async function () {
	
	
	fileList = $("#image-selector").prop('files');
	
	//$("#prediction-list").empty();
	
	model_processArray(fileList);
	
});






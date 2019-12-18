// UI unit
var record_btn;
var canvas;
var interval_key;

// javascript object
var mContext = null;
var mAudioInput = null;
var mRecorder = null;

// constants
var sampleRate = 0;
var forcestop = false;
var animationId = null;
var WORKER_PATH = 'recorderWorker.js';




Array.prototype.reshape = function(rows, cols) {
  var copy = this.slice(0); // Copy all elements.
  this.length = 0; // Clear out existing array.

  for (var r = 0; r < rows; r++) {
    var row = [];
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (i < copy.length) {
        row.push(copy[i]);
      }
    }
    this.push(row);
  }
};

/************************************ 로드 후 실행 ******************************************/
function onLoad(){
    record_btn = document.getElementById("record");
    // 오디오 모듈 만들고, 디바이스 얻고, 연결 구성
    makeAudioContext().then(getDevices).then(startUserMedia);
    // minimal heatmap instance configuration
}

// 버튼 클릭
var toggleRecording = function(e) {
    if (e.classList.contains("recording")) {

	forcestop = true;
	on_stop();
    } else {

	forcestop= false;
	on_record();
    }
};


/****************************** AUDIO MODULE ************************************/

// 오디오 모듈 만들기
function makeAudioContext(){
    return new Promise(function (resolve, reject){
        try {
        window.AudioContext = window.AudioContext || window.webkitAudioContext;
        mContext = new AudioContext();
        window.URL = window.URL || window.webkitURL;
        resolve();

    } catch (e) {
        throw 'WebAudio API has no support on this browser.';
        console.log( 'WebAudio API has no support on this browser.');
    }

    });
}

// 오디오 디바이스 얻기
function getDevices(){
        if(navigator.mediaDevices){
            /** * source Constratints ** */
            var audioSourceConstraints = {
                audio: {optional: [{echoCancellation:false}]},
                video: false
            };
            return navigator.mediaDevices.getUserMedia(audioSourceConstraints);

        } else console.log("No Devices");
}

// 오디오 스트림 연결 구성
function startUserMedia(stream){
    mAudioInput = mContext.createMediaStreamSource(stream);
    // Firefox를 위한 설정 : 5초마다 오디오 스트림이 끊기지 않게.
    window.source = mAudioInput;

   mAudioInput.connect(mContext.destination);

    mRecorder = new Recorder(mAudioInput, { workerPath : './static/lib/recorderWorker.js' });
}

/****************************** AUDIO MODULE ************************************/

// 녹음
var on_record = function(){
    function keepgoing() {
	if(mContext.state === 'suspended') {
	    console.log('state ==> resume')
	    mContext.resume();
	}
	var temp = document.getElementById("record");
	if(!(temp.classList.contains("recording"))) {
	    temp.classList.add("recording");
	    mRecorder.allclear();
	    mRecorder.record();
            document.getElementById("message").innerHTML = "";
            document.getElementById("score").innerHTML = "";
	}
    } // keepgoing
    window.setTimeout(keepgoing,300);
    interval_key = window.setInterval(function(){
    	mRecorder.getFullbuffer(gotBuffers);
    }, 100);

}

// 오디오 정지
var on_stop = function() {

    //그림 정지
    clearInterval(interval_key);
    var temp = document.getElementById("record");
    if(temp.classList.contains("recording")) {
	// 녹음중이면
	if(mContext.state === 'running') {
	    console.log('state ==> resume')
	    mContext.suspend();
	}
	temp.classList.remove("recording");
	mRecorder.stop();
	console.log("record stop");
        mRecorder.export16kMono(function(blob){
            sendToServer(blob);
            mRecorder.clear();
        });
    }
}

/****************************** ??? ************************************/
function sendToServer(blob){
    var filename = new Date().toISOString();
    var req = new XMLHttpRequest();
    req.responseType = 'json';
    req.onload = function(data){
        // var data = req.response;
        onResponse(data);

    };
    // var fd=new FormData();
    // fd.append("audio_data",blob, filename);
    req.open('POST', '/', true);
    req.send(blob);
}

// make table of audio as the server responded
function onResponse(data){
    console.log(data.srcElement.response);
    var response = data.srcElement.response;
    if(response.event == "no_result"){
	document.getElementById("message").innerHTML ="Error occurred: please try again";
    }
    else if (response.event=="result") {
        console.log('result received');
        var payload = JSON.stringify(response.payload);
        var textMessage = JSON.parse(payload);
        var spk_list = textMessage.result;

        console.log(spk_list);
        var table_str='<tr><th>가장 가까운 목소리</th></tr>';
        for(var i=0; i<spk_list.length; i++){
            table_str += '<tr><td><audio controls id="' +spk_list[i]+ '"></audio></td></tr>';
        }
        table_str = '<table>'+table_str+'</table>';
        
        document.getElementById('score').innerHTML = table_str;
        var audElems = document.getElementsByTagName('audio');
        for(var i=0;i<audElems.length;i++){
            getwav(audElems[i].id, audElems[i]);
        }
    }
}

// get wavs from server
function getwav(wavpath, audObj){
    var req = new XMLHttpRequest();
    req.open('POST', '/wav', true);

    req.responseType = 'blob';
    req.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    req.onload = function(evt){
	var blob = new Blob([req.response], {type: 'audio/mp4'});
	var objectUrl = URL.createObjectURL(blob);
	audObj.src = objectUrl;
	audObj.onload = function(evt) {
	    URL.revokeObjectURL(objectUrl);
	};
    };
    req.send(JSON.stringify({"path":wavpath}));
}

/****************************** DRAW CANVAS ************************************/

// 오디오 콜백 받기 함수
var createDownloadLink = function() {
		mRecorder.exportWAV(function(blob) {
			if (blob.size > 6000) {
				//  오디오 콜백 받기
				mRecorder.getFullbuffer(gotBuffers);
			}
		})
	}

// 캔버스 그림 그리기
var gotBuffers = function(buffers) {
		// console.log(buffers[0])

		canvas = document.getElementById("wavedisplay");
		drawBuffer( canvas.width, canvas.height, canvas.getContext('2d'), buffers[0] );
}

// 그림 그리기 함수
var drawBuffer = function( width, height, context, data ) {
	    var step = Math.ceil( data.length / width );
	    var amp = height / 2;
	    context.fillStyle = "silver";
	    context.clearRect(0,0,width,height);
	    for(var i=0; i < width; i++){
	        var min = 1.0;
	        var max = -1.0;
	        for (j=0; j<step; j++) {
	            var datum = data[(i*step)+j];
	            if (datum < min)
	                min = datum;
	            if (datum > max)
	                max = datum;
	        }
	        context.fillRect(i,(1+min)*amp,1,Math.max(1,(max-min)*amp));
	    }
	}
/****************************** Recorder ************************************/
var Recorder = function(source, cfg){
	var config = cfg || {};
	var bufferLen = config.bufferLen || 4096;
	this.context = source.context;
	this.node = this.context.createScriptProcessor(bufferLen, 1, 1);
	var worker = new Worker(config.workerPath || WORKER_PATH);
	worker.postMessage({
		command: 'init',
		config: {
			sampleRate: this.context.sampleRate
		}
	});
	var recording = false,
	currCallback;
	this.node.onaudioprocess = function(e){
		//  버퍼 콜백
		if (!recording) return;
		var buffer = e.inputBuffer.getChannelData(0);
		worker.postMessage({
			command: 'record',
			buffer: [buffer]
		});
	}
	this.configure = function(cfg){
		for (var prop in cfg){
			if (cfg.hasOwnProperty(prop)){
				config[prop] = cfg[prop];
			}
		}
	}
	this.record = function(){
		recording = true;
	}
	this.stop = function(){
		recording = false;
	}
	this.clear = function(){
		worker.postMessage({ command: 'clear' });
	}
	this.allclear = function(){
		worker.postMessage({ command: 'allclear' });
	}
	this.getBuffer = function(cb) {
		currCallback = cb || config.callback;
		worker.postMessage({ command: 'getBuffer' })
	}
	this.getFullbuffer = function(cb) {
		currCallback = cb || config.callback;
		worker.postMessage({ command: 'getFullbuffer' })
	}
	this.getFullLength = function() {
		worker.postMessage({ command: 'getFullLength' })
	}
	this.exportWAV = function(cb, type){
		currCallback = cb || config.callback;
		type = type || config.type || 'audio/wav';
		if (!currCallback) throw new Error('Callback not set');
		worker.postMessage({
			command: 'exportWAV',
			type: type
		});
	}
	this.exportRAW = function(cb, type){
		currCallback = cb || config.callback;
		type = type || config.type || 'audio/raw';
		if (!currCallback) throw new Error('Callback not set');
		worker.postMessage({
			command: 'exportRAW',
			type: type
		});
	}
	this.export16kMono = function(cb, type){
		currCallback = cb || config.callback;
		type = type || config.type || 'audio/raw';
		if (!currCallback) throw new Error('Callback not set');
		worker.postMessage({
			command: 'export16kMono',
			type: type
		});
	}
	// FIXME: doesn't work yet
	this.exportSpeex = function(cb, type){
		currCallback = cb || config.callback;
		type = type || config.type || 'audio/speex';
		if (!currCallback) throw new Error('Callback not set');
		worker.postMessage({
			command: 'exportSpeex',
			type: type});
	}
	worker.onmessage = function(e){
		var blob = e.data;
		currCallback(blob);
	}
	source.connect(this.node);
	this.node.connect(this.context.destination);
};

/**************************** 실행부 *****************************/

(function(window){

	Recorder.forceDownload = function(blob, filename){
		var url = (window.URL || window.webkitURL).createObjectURL(blob);
		var link = window.document.createElement('a');
		link.href = url;
		link.download = filename || 'output.wav';
		var click = document.createEvent("Event");
		click.initEvent("click", true, true);
		link.dispatchEvent(click);
	}

	window.Recorder = Recorder;

})(window);

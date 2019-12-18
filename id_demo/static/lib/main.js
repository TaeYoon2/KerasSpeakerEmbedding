//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext; //audio context to help us record

var recordButton = document.getElementById("recordButton");

//add events to those 2 buttons
// recordButton.addEventListener("click", startRecording);

function startRecording() {
    var constraints = { audio: true, video:false };
    
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
	console.log("getUserMedia() success, stream created, initializing Recorder.js ...");
	audioContext = new AudioContext();

	/*  assign to gumStream for later use  */
	gumStream = stream;
		
	/* use the stream */
	input = audioContext.createMediaStreamSource(stream);

	/* 
	   Create the Recorder object and configure to record mono sound (1 channel)
	   Recording 2 channels  will double the file size
	*/
	rec = new Recorder(input,{numChannels:1});
	rec.record();

	console.log("Recording started");
        
    }).catch(function(err) {
	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    });
}



function stopRecording() {
	console.log("stopButton clicked");
	rec.stop();

	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {
    
	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');

	//name of .wav file to use during upload and download (without extendion)
	var filename = new Date().toISOString();

	//add controls to the <audio> element
	au.controls = true;
	au.src = url;

	//save to disk link
	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
}


function sendToServer(event){
    var xhr=new XMLHttpRequest();
    xhr.onload=function(e) {
	if(this.readyState === 4) {
	    console.log("Server returned: ",e.target.responseText);
	}
    };
    var fd=new FormData();
    fd.append("audio_data",blob, filename);
    xhr.open("POST","upload.php",true);
    xhr.send(fd);
}


function onResponse(data){
    var payload = JSON.stringify(data.payload);
    if(data.event == "no_result"){
	writeToScreen("Error occurred: please try again");
    }
    if (data.event=="result") {
	var textMessage = JSON.parse(payload);

	var res_dict = textMessage.result;
	var closest = res_dict['closest_lab'];
	var close_sim = res_dict['closest_sim'];

	var table_str='<tr><th>가장 가까운 목소리</th><th>코사인 유사도</th></tr>';
    	for(var i=0; i<closest.length; i++){
    	    table_str += '<tr><td><audio controls id="' +closest[i]+ '"></audio></td><td>' +close_sim[i]+ '</td></tr>';
    	}
	console.log(table_str);
	table_str = '<table>'+table_str+'</table>';
	sgvdiv.innerHTML =table_str;

        var audElems = document.getElementsByTagName('audio');
        for(i=0;i<audElems.length;i++){	
	    getwav(audElems[i].id, audElems[i]);
        }
    }
}


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

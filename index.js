const recordButton = document.getElementById("recordButton");
const status = document.getElementById("status");
const transcription = document.getElementById("transcription");

let mediaRecorder;
let audioChunks = [];


if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
  transcription.innerText = "Speech recognition not supported in this browser.";
  recordButton.disabled = true;
}

recordButton.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordButton.innerText = "Start Recording";
    status.innerText = "Stopped recording...";
  } else {
    startRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });
    const audioContext = new AudioContext();
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.onstart = () => {
      audioChunks = [];
      recordButton.innerText = "Stop Recording";
      status.innerText = "Recording...";
    };

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const audioURL = URL.createObjectURL(audioBlob);

      const recognition = new (window.SpeechRecognition ||
        window.webkitSpeechRecognition)();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        console.log("Speech recognition service has started");
      };

      recognition.onresult = async (event) => {
        const transcript = event.results[0][0].transcript;
        console.log(`Transcript: ${transcript}`);

        try {
          const response = await fetch("/transcribe", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: transcript }),
          });

          if (!response.ok)
            throw new Error(`Server responded with status ${response.status}`);
          const result = await response.json();

          if (result.error) throw new Error(result.error);

          transcription.innerText =
            result.isl_translation.join(" ") || "No translation available";
        } catch (error) {
          transcription.innerText = `Error: ${error.message}`;
        }
      };

      recognition.onerror = (event) => {
        console.error(`Speech recognition error: ${event.error}`);
        transcription.innerText = `Error: ${event.error}`;
      };

      recognition.onend = () => {
        status.innerText = "Speech recognition ended.";
      };

      recognition.start();
    };

    mediaRecorder.start();
  } catch (error) {
    console.error(`Recording error: ${error.message}`);
    status.innerText = `Error: ${error.message}`;
  }
}

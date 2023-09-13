function getPrediction() {
    const fileInput = document.getElementById('fileInput');
    const outputFormat = document.getElementById('outputFormat').value;

    if (!fileInput.files.length) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('dataFile', fileInput.files[0]);
    formData.append('format', outputFormat);

    // Send a POST request to the Flask API with the file and desired format
    let url='http://localhost:8080/predict/audiogenev4'
    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (outputFormat === 'json') {
            return response.json();
        } else {
            return response.text();
        }
    })
    .then(data => {
        document.getElementById('result').innerText = `Prediction: ${data}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

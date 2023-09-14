from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import subprocess
import os
import docker
app = Flask(__name__)

docker_client = docker.from_env()
SHARED_DIR = '/' + os.getenv('SHARED_DATA')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/audiogenev4', methods=['POST'])
def predict_audiogenev4():
    # Check if a file was uploaded
    if 'dataFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dataFile']
    
    # Check if the file is empty
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size == 0:
        return jsonify({'error': 'Uploaded file is empty'}), 400
    
    # Save the uploaded file to a temporary location
    filepath = os.path.join(SHARED_DIR, file.filename)
    try:
        file.save(filepath)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file'}), 500
    
    base_service = '/audiogene-web-service-'
    file_path = os.path.splitext(filepath)[0]
    
    # Get convert the input file to a csv
    csv_file_path = f"{file_path}.csv"
    container = get_container_by_name(docker_client, base_service+'xlstocsv-1')
    cmd = f'java -jar ConvertToXLS2CSV.jar {filepath} {csv_file_path}'
    container.exec_run(cmd)
    
    # Apply perl preprocessing
    post_processing_path = f"{file_path}_processed.out"
    container = get_container_by_name(docker_client, base_service+'perl_preprocessor-1')
    cmd = f'perl weka-preprocessor.perl -i -a -poly=3 {csv_file_path} > {post_processing_path}'
    res = container.exec_run(cmd).output
    with open(post_processing_path, 'wb') as f:
        f.write(res)
    
    # Send processed file to the classifier
    predictions_path = f"{file_path}_predictions.csv"
    container = get_container_by_name(docker_client, base_service+'audiogenev4-1')
    cmd = f'java -jar -Xmx2G AudioGene.jar {post_processing_path} audiogene.misvm.model > {predictions_path}'
    res = container.exec_run(cmd).output
    
    # Save predictions to a file
    with open(predictions_path, 'wb') as f:
        f.write(res.replace(b'\t', b',')) # Replace the sporadic tab seperation with comma seperation
    
    # Read the result using pandas
    data = pd.read_csv(predictions_path, index_col=0)
    data.columns = [x for x in range(1, data.shape[1]+1)]

    outputFormat = request.form.get('format', 'json')
    # Return the result in the desired format
    if outputFormat == 'json':
        return jsonify(data.to_dict(orient='index')), 200
    elif outputFormat == 'csv':
        csv_data = data.to_csv(index=False)
        return csv_data, 200, {'Content-Type': 'text/csv'}
    else:
        return jsonify({'error': 'Unsupported output format'}), 400    

@app.route('/predict/audiogenev9', methods=['POST'])
def predict_audiogenev9():
    # Check if a file was uploaded
    if 'dataFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dataFile']
    
    # Check if the file is empty
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size == 0:
        return jsonify({'error': 'Uploaded file is empty'}), 400
    
    # Save the uploaded file to a temporary location
    filepath = os.path.join(SHARED_DIR, file.filename)
    try:
        file.save(filepath)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file'}), 500
    
    base_service = '/audiogene-web-service-'
    file_path = os.path.splitext(filepath)[0]
    
    o = '/shared_data/ag9_output.csv'
    m = './audiogene_ml/audiogene-ml/notebooks/saved_models/cur_model_lv3_compression.joblib'
    i = file_path
    
    cmd = f'python ./audiogene_ml/audiogene-ml/predict.py -i {i} -o {o} -m {m}'
    container = get_container_by_name(docker_client, base_service+'audiogenev9-1')
    container.exec_run(cmd)
    
    # Read the result using pandas
    data = pd.read_csv(o, index_col=0)

    outputFormat = request.form.get('format', 'json')
    # Return the result in the desired format
    if outputFormat == 'json':
        return jsonify(data.to_dict(orient='index')), 200
    elif outputFormat == 'csv':
        csv_data = data.to_csv(index=False)
        return csv_data, 200, {'Content-Type': 'text/csv'}
    else:
        return jsonify({'error': 'Unsupported output format'}), 400

def get_container_by_name(client:docker.client.DockerClient, name:str) -> docker.models.containers.Container:
    containers = client.containers.list()
    container_by_name = {c.attrs['Name']:c.attrs['Id'] for c in containers}
    if name in container_by_name.keys():
        container_id = container_by_name[name]
        return client.containers.get(container_id)
    else:
        raise ValueError(f'could not find container {name}')

# Only executes when running locally OUTSIDE of Docker container
# __name__ will be 'app' when executing from the container
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=os.getenv('API_INTERNAL_PORT'))
    app.run(host=api_host, port=internal_api_port)

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import subprocess
import os
import docker
import logging

app = Flask(__name__)

docker_client = docker.from_env()
SHARED_DIR = '/' + os.getenv('SHARED_DATA', 'shared_data')

# Get the directory for this file
path = os.path.dirname(os.path.abspath(__file__))
# Get the filepath for the ./test.xlsx file
# print the current working directory
print("Current working directory: ", os.getcwd())
# print the contentes of the current working directory
print("Current working directory contents: ", os.listdir())
# filepath_test = os.path.join(path, 'test.xlsx')
# # Copy the file to the shared directory
# subprocess.run(['cp', filepath_test, SHARED_DIR])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_curl', methods=['POST'])
def test_curl():
    return jsonify({'message': 'success'})

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
    print("csv file path: ", csv_file_path)
    container = get_container_by_name(docker_client, base_service+'xlstocsv-1')
    cmd = f'java -jar ConvertToXLS2CSV.jar {filepath} {csv_file_path}'
    # convert xls to csv using python
    # import pandas as pd
    # data_xls = pd.read_excel(filepath, index_col=None)
    # data_xls.to_csv(csv_file_path, encoding='utf-8', index=False)
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
    
    # Read the first line of the file
    with open(predictions_path, 'r') as f:
        first_line = f.readline().strip()

    # Check if the first line contains 'ID' followed by a number
    import re
    if re.match(r'ID \d+', first_line):
        # If it does, read the file without a header
        data = pd.read_csv(predictions_path, header=None)
    else:
        # Otherwise, read the file normally
        data = pd.read_csv(predictions_path)

    # Rename the columns
    print("Data read from csv file after predictions: ", data)
    data.columns = [x for x in range(1, data.shape[1]+1)]
    print("Data after column renaming: ", data)

    outputFormat = request.form.get('format', 'json')
    # Return the result in the desired format
    if outputFormat == 'json':
        print("returning json: {}".format(data.to_dict(orient='index')))
        return jsonify(data.to_dict(orient='index')), 200
    elif outputFormat == 'csv':
        csv_data = data.to_csv(index=False)
        return csv_data, 200, {'Content-Type': 'text/csv'}
    else:
        return jsonify({'error': 'Unsupported output format'}), 400    

@app.route('/predict/audiogenev9', methods=['POST'])
def predict_audiogenev9():
    try:
        print('predicting audigenev9')
        logging.info('predicting audigenev9')
        # Check if a file was uploaded
        if 'dataFile' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['dataFile']
        print(file)
        
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
        
        o = os.path.join(SHARED_DIR, 'ag9_output.csv')
        # Save the output to a temporary location
        # Create an empty csv file
        open(o, 'a').close()

        m = './audiogene_ml/audiogene-ml/notebooks/saved_models/cur_model_lv3_compression.joblib'
        i = filepath
        
        cmd = f'python -u ./audiogene_ml/audiogene-ml/predict.py -i {i} -o {o} -m {m}'
        print("running command: ", cmd)
        container = get_container_by_name(docker_client, base_service+'audiogenev9-1')
        container.exec_run(cmd)

        print("finished running command")
        
        # Read the result using pandas
        data = pd.read_csv(o, index_col=0)

        print(data)

        outputFormat = request.form.get('format', 'json')
        # Return the result in the desired format
        if outputFormat == 'json':
            return jsonify(data.to_dict(orient='index')), 200
        elif outputFormat == 'csv':
            csv_data = data.to_csv(index=False)
            return csv_data, 200, {'Content-Type': 'text/csv'}
        else:
            return jsonify({'error': 'Unsupported output format'}), 400
    except Exception as e:
        print(e)
        return jsonify({'error': f'Error predicting{e}'}), 500

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
    app.run(host='0.0.0.0', port=os.getenv('API_INTERNAL_PORT', 5001), debug=True)
    # app.run(host=api_host, port=internal_api_port, debug=True)

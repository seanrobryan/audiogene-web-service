from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import subprocess
import os
import docker
import logging

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# docker_client = docker.from_env()
SHARED_DIR = '/' + 'shared_data'

# Get the directory for this file
path = os.path.dirname(os.path.abspath(__file__))
# shared_path = os.path.join(path, 'shared_data')
shared_path = SHARED_DIR

# CSV_FILE_PATH = 'ag_9_full_dataset_processed_with_var.csv'
CSV_FILE_PATH = os.path.join(path, 'ag_9_full_dataset_processed_with_var.csv')

GENES = ['ACTG1', 'CCDC50', 'CEACAM16', 'COCH', 'COL11A2', 'EYA4', 'DIAPH1',
         'GJB2', 'GRHL2', 'GSDME', 'KCNQ4', 'MIRN96', 'MYH14', 'MYH9', 'MYO6',
         'MYO7A', 'P2RX2', 'POU4F3', 'REST', 'SLC17A8', 'TECTA', 'TMC1', 'WFS1']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test_curl', methods=['POST'])
def test_curl():
    return jsonify({'message': 'success'})


@app.route('/data_visualization', methods=['GET'])
def data_visualization():
    df = pd.read_csv(CSV_FILE_PATH)

    # Define a dictionary mapping the old column names to the new ones
    column_mapping = {
    '125 dB': '125 Hz',
    '250 dB': '250 Hz',
    '500 dB': '500 Hz',
    '1000 dB': '1000 Hz',
    '1500 dB': '1500 Hz',
    '2000 dB': '2000 Hz',
    '3000 dB': '3000 Hz',
    '4000 dB': '4000 Hz',
    '6000 dB': '6000 Hz',
    '8000 dB': '8000 Hz'
    }

    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)

    # Ensure that 'ethnicity' is excluded from the numeric conversion process
    features = ['age', '125 Hz', '250 Hz', '500 Hz', '1000 Hz', '1500 Hz', '2000 Hz', '3000 Hz', '4000 Hz', '6000 Hz', '8000 Hz']
    target = 'gene'

    # Preprocess the data
    processed_data = df[features + [target, 'ethnicity']].dropna()  # Include 'ethnicity' for later encoding

    # Convert all feature columns to numeric, coercing errors to NaN, then drop rows with NaN
    for feature in features:
        processed_data[feature] = pd.to_numeric(processed_data[feature], errors='coerce')
    processed_data.dropna(inplace=True)

    # Encode the gene and ethnicity
    label_encoder_gene = LabelEncoder()
    processed_data['gene_encoded'] = label_encoder_gene.fit_transform(processed_data[target])
    gene_labels = label_encoder_gene.classes_  # Save the original gene labels

    label_encoder_ethnicity = LabelEncoder()
    processed_data['ethnicity_encoded'] = label_encoder_ethnicity.fit_transform(processed_data['ethnicity'])

   # Data for 3D Clustering
    pca = PCA(n_components=3)
    clustering_features = pca.fit_transform(processed_data[features])
    kmeans = KMeans(n_clusters=23)  # Adjust the number of clusters as needed
    clustering_labels = kmeans.fit_predict(clustering_features)

    # Include gene information in the DataFrame from the start
    clustering_data = pd.DataFrame({
        'x': clustering_features[:, 0],
        'y': clustering_features[:, 1],
        'z': clustering_features[:, 2],
        'cluster': clustering_labels,
        'gene': processed_data['gene'].values
    })

    # Create a dictionary of genes in each cluster
    genes_in_clusters = clustering_data.groupby('cluster')['gene'].apply(list).to_dict()


    # Create a dictionary of unique genes in each cluster
    unique_genes_in_clusters = {cluster: list(set(genes)) for cluster, genes in genes_in_clusters.items()}


    # Data for Ethnicity vs Gene Scatter Plot
    scatter_data = processed_data[['ethnicity', 'gene']].copy()
    scatter_data = scatter_data.rename(columns={'ethnicity': 'x', 'gene': 'y', 'ethnicity_encoded': 'x_encoded',
                                                'gene_encoded': 'y_encoded'})

    # Data for Gene Expression Over Age
    gene_expression_data = processed_data[['age', 'gene_encoded']].copy()
    gene_expression_data['gene'] = gene_expression_data['gene_encoded'].apply(lambda x: gene_labels[x])

    heatmap = processed_data[features + ['gene_encoded', 'ethnicity_encoded']].corr()
    heatmap_data = heatmap.to_dict(orient='index')

    # Data for Audiograms Per Gene
    audiograms_per_gene = df['gene'].value_counts().to_dict()

    # Data for Hearing Loss Over Age
    hearing_loss_over_age = processed_data.groupby('age', as_index=False)[features].mean()
    hearing_loss_over_age = hearing_loss_over_age.to_dict(orient='records')

    # Features without age
    features_without_age = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '1500 Hz', '2000 Hz', '3000 Hz', '4000 Hz', '6000 Hz', '8000 Hz']

    # Data for Hearing Loss by Frequency
    hearing_loss_by_frequency = {feature: processed_data[feature].mean() for feature in features_without_age}

    # Data for Hearing Loss Distribution
    hearing_loss_distribution = processed_data[features].values.flatten()
    hearing_loss_distribution = pd.Series(hearing_loss_distribution).value_counts().to_dict()

    # Data for Ethnicity Distribution
    ethnicity_distribution = processed_data['ethnicity'].value_counts().to_dict()

    # Combine all the data into a single JSON object
    visualization_data = {
        'heatmapData': heatmap_data,
        'uniqueGenesInClusters': unique_genes_in_clusters,
        'clusteringData': clustering_data.to_dict(orient='records'),
        'scatterData': scatter_data.to_dict(orient='records'),
        'ageData': processed_data['age'].tolist(),
        'geneExpressionData': gene_expression_data.to_dict(orient='records'),
        'audiogramsPerGene': audiograms_per_gene,
        'hearingLossOverAge': hearing_loss_over_age,
        'hearingLossByFrequency': hearing_loss_by_frequency,
        'hearingLossDistribution': hearing_loss_distribution,
        'ethnicityDistribution': ethnicity_distribution,

    }

    print("Completed data visualization")

    return jsonify(visualization_data)

# @app.route('/predict/audiogenev4', methods=['POST'])
# def predict_audiogenev4():
#     try:
#         # Check if a file was uploaded
#         if 'dataFile' not in request.files:
#             return jsonify({'error': 'No file uploaded'}), 400
#
#         file = request.files['dataFile']
#
#         # Check if the file is empty
#         file.seek(0, os.SEEK_END)
#         size = file.tell()
#         file.seek(0)
#         if size == 0:
#             return jsonify({'error': 'Uploaded file is empty'}), 400
#
#         # Save the uploaded file to a temporary location
#         filepath = os.path.join(shared_path, file.filename)
#         try:
#             file.save(filepath)
#         except Exception as e:
#             print(f"Error saving file: {e}")
#             return jsonify({'error': f'Error saving file: {e}'}), 500
#
#         base_service = '/audiogene-web-service-'
#         file_path = os.path.splitext(filepath)[0]
#
#         # Get convert the input file to a csv
#         csv_file_path = f"{file_path}.csv"
#         print("csv file path: ", csv_file_path)
#         container = get_container_by_name(docker_client, base_service + 'xlstocsv-1')
#         cmd = f'java -jar ConvertToXLS2CSV.jar {filepath} {csv_file_path}'
#         # convert xls to csv using python
#         # import pandas as pd
#         # data_xls = pd.read_excel(filepath, index_col=None)
#         # data_xls.to_csv(csv_file_path, encoding='utf-8', index=False)
#         container.exec_run(cmd)
#
#         # Apply perl preprocessing
#         post_processing_path = f"{file_path}_processed.out"
#         container = get_container_by_name(docker_client, base_service + 'perl_preprocessor-1')
#         cmd = f'perl weka-preprocessor.perl -i -a -poly=3 {csv_file_path} > {post_processing_path}'
#         res = container.exec_run(cmd).output
#         with open(post_processing_path, 'wb') as f:
#             f.write(res)
#
#         # Send processed file to the classifier
#         predictions_path = f"{file_path}_predictions.csv"
#         container = get_container_by_name(docker_client, base_service + 'audiogenev4-1')
#         cmd = f'java -jar -Xmx2G AudioGene.jar {post_processing_path} audiogene.misvm.model > {predictions_path}'
#         res = container.exec_run(cmd).output
#
#         # Save predictions to a file
#         with open(predictions_path, 'wb') as f:
#             f.write(res.replace(b'\t', b','))  # Replace the sporadic tab seperation with comma seperation
#
#         # Read the first line of the file
#         with open(predictions_path, 'r') as f:
#             first_line = f.readline().strip()
#
#         # Check if the first line contains 'ID' followed by a number
#         import re
#         if re.match(r'ID \d+', first_line):
#             # If it does, read the file without a header
#             data = pd.read_csv(predictions_path, header=None)
#         else:
#             # Otherwise, read the file normally
#             data = pd.read_csv(predictions_path)
#
#         # Rename the columns
#         print("Data read from csv file after predictions: ", data)
#         data.columns = [x for x in range(1, data.shape[1] + 1)]
#         print("Data after column renaming: ", data)
#
#         outputFormat = request.form.get('format', 'json')
#         # Return the result in the desired format
#         if outputFormat == 'json':
#             print("returning json: {}".format(data.to_dict(orient='index')))
#             return jsonify(data.to_dict(orient='index')), 200
#         elif outputFormat == 'csv':
#             csv_data = data.to_csv(index=False)
#             return csv_data, 200, {'Content-Type': 'text/csv'}
#         else:
#             return jsonify({'error': 'Unsupported output format'}), 400
#     except Exception as e:
#         print(e)
#         return jsonify({'error': f'Error predicting{e}'}), 500
#
#
# @app.route('/predict/audiogenev9', methods=['POST'])
# def predict_audiogenev9():
#     try:
#         print('predicting audigenev9')
#         logging.info('predicting audigenev9')
#         # Check if a file was uploaded
#         if 'dataFile' not in request.files:
#             return jsonify({'error': 'No file uploaded'}), 400
#
#         file = request.files['dataFile']
#         print(file)
#
#         # Check if the file is empty
#         file.seek(0, os.SEEK_END)
#         size = file.tell()
#         file.seek(0)
#         if size == 0:
#             return jsonify({'error': 'Uploaded file is empty'}), 400
#
#         # Save the uploaded file to a temporary location
#         # filepath = os.path.join(SHARED_DIR, file.filename)
#         filepath = os.path.join(shared_path, file.filename)
#         # check if shared dir exists if not create it
#         if not os.path.exists(shared_path):
#             os.makedirs(shared_path)
#         try:
#             file.save(filepath)
#         except Exception as e:
#             print(f"Error saving file: {e}")
#             return jsonify({'error': f'Error saving file to share_dir: {e}'}), 404
#
#         base_service = '/audiogene-web-service-'
#         file_path = os.path.splitext(filepath)[0]
#
#         o = os.path.join(shared_path, 'ag9_output.csv')
#         # Save the output to a temporary location
#         # Create an empty csv file
#         open(o, 'a').close()
#
#         m = './audiogene_ml/audiogene-ml/notebooks/saved_models/cur_model_lv3_compression.joblib'
#         i = filepath
#
#         cmd = f'python -u ./audiogene_ml/audiogene-ml/predict.py -i {i} -o {o} -m {m}'
#         print("running command: ", cmd)
#         container = get_container_by_name(docker_client, base_service + 'audiogenev9-1')
#         container.exec_run(cmd)
#
#         print("finished running command")
#
#         # Read the result using pandas
#         data = pd.read_csv(o, index_col=0)
#
#         print(data)
#
#         outputFormat = request.form.get('format', 'json')
#         # Return the result in the desired format
#         if outputFormat == 'json':
#             print("returning json")
#             return jsonify(data.to_dict(orient='index')), 200
#         elif outputFormat == 'csv':
#             print("returning csv")
#             csv_data = data.to_csv(index=False)
#             return csv_data, 200, {'Content-Type': 'text/csv'}
#         else:
#             print("returning error")
#             return jsonify({'error': 'Unsupported output format'}), 400
#     except Exception as e:
#         print(e)
#         return jsonify({'error': f'Error predicting{e}'}), 500


def get_container_by_name(client: docker.client.DockerClient, name: str) -> docker.models.containers.Container:
    containers = client.containers.list()
    container_by_name = {c.attrs['Name']: c.attrs['Id'] for c in containers}
    if name in container_by_name.keys():
        container_id = container_by_name[name]
        return client.containers.get(container_id)
    else:
        raise ValueError(f'could not find container {name}')


# Only executes when running locally OUTSIDE of Docker container
# __name__ will be 'app' when executing from the container
if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    # app.run(host=api_host, port=internal_api_port, debug=True)

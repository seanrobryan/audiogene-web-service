from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy.stats import zscore
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import subprocess
import os
import docker
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


app = Flask(__name__)

# docker_client = docker.from_env()
SHARED_DIR = '/' + 'shared_data'

docker_client = docker.from_env()

# Get the directory for this file
path = os.path.dirname(os.path.abspath(__file__))
# shared_path = os.path.join(path, 'shared_data')
shared_path = SHARED_DIR

# CSV_FILE_PATH = 'ag_9_full_dataset_processed_with_var.csv'
CSV_FILE_PATH = os.path.join(path, 'ag_9_full_dataset_processed_test_no_added_data.csv')

GENES = ['ACTG1', 'CCDC50', 'CEACAM16', 'COCH', 'COL11A2', 'EYA4', 'DIAPH1',
         'GJB2', 'GRHL2', 'GSDME', 'KCNQ4', 'MIRN96', 'MYH14', 'MYH9', 'MYO6',
         'MYO7A', 'P2RX2', 'POU4F3', 'REST', 'SLC17A8', 'TECTA', 'TMC1', 'WFS1']
age_groups = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float('inf'))]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test_curl', methods=['POST'])
def test_curl():
    return jsonify({'message': 'success'})


# [2] Refined Outlier Removal (Using Z-score)
def remove_outliers_zscore_per_age_group_and_gene(df, threshold=3, underrepresented_threshold=35):
    # Define age groups

    # Identifying underrepresented genes
    underrepresented_genes = df['gene'].value_counts()[df['gene'].value_counts() < underrepresented_threshold].index
    df_underrepresented = df[df['gene'].isin(underrepresented_genes)]
    df_others = df[~df['gene'].isin(underrepresented_genes)]

    # List to store DataFrame segments after outlier removal
    cleaned_data_segments = []

    for gene in df_others['gene'].unique():
        df_gene = df_others[df_others['gene'] == gene]

        for age_group in age_groups:
            # Filter data for the current age group
            df_gene_age_group = df_gene[(df_gene['age'] >= age_group[0]) & (df_gene['age'] < age_group[1])]

            # Skip empty datasets
            if df_gene_age_group.empty:
                continue

            # Apply Z-score based outlier removal
            numeric_data = df_gene_age_group.select_dtypes(include=[np.number])
            z_scores = np.abs(zscore(numeric_data))
            df_gene_age_group_clean = df_gene_age_group[(z_scores < threshold).all(axis=1)]

            # Add the cleaned segment to the list
            cleaned_data_segments.append(df_gene_age_group_clean)

    # Concatenate all cleaned segments and underrepresented genes into a single DataFrame
    cleaned_df = pd.concat(cleaned_data_segments + [df_underrepresented], ignore_index=True)
    return cleaned_df


def dynamic_class_balancing(df, feature_columns, target_column, label_encoder_gene, label_encoder_ethnicity):
    # Define the target and features
    target = df['gene']
    feature_columns = feature_columns + ['ethnicity']
    features = df[feature_columns]
    feature_columns = feature_columns[:-1]
    # Create a unique key for each original sample by summing up the frequency labels and appending the gene
    df['unique_key'] = df[feature_columns].sum(axis=1).astype(str) + '_' + df['gene'].astype(str)
    original_keys = df['unique_key'].values
    original_patient_id_mapping = df.set_index('unique_key')['patient_id_family_id'].to_dict()

    # Encode target labels
    le_gene = label_encoder_gene
    y_encoded = le_gene.fit_transform(target)

    # Encode feature labels 'ethnicity'
    label_encoder_ethnicity = label_encoder_ethnicity
    ethnicity_encoded = label_encoder_ethnicity.fit_transform(df['ethnicity'])
    features = features.drop(['ethnicity'], axis=1)
    features['ethnicity'] = ethnicity_encoded

    # Define dynamic sampling strategy for SMOTE and NearMiss
    class_counts = np.bincount(y_encoded)
    smote_strategy = {i: 50 for i, count in enumerate(class_counts) if count < 50}
    nm_strategy = {i: 200 for i, count in enumerate(class_counts) if count > 200}

    # Apply SMOTE for oversampling and NearMiss for undersampling
    smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=2)
    nm = NearMiss(sampling_strategy=nm_strategy)
    X_res, y_res = smote.fit_resample(features, y_encoded)
    X_res, y_res = nm.fit_resample(X_res, y_res)

    # Reconstruct DataFrame
    df_resampled = pd.DataFrame(X_res, columns=features.columns)
    df_resampled['gene'] = le_gene.inverse_transform(y_res)
    df_resampled['unique_key'] = df_resampled[feature_columns].sum(axis=1).astype(str) + '_' + df_resampled['gene']

    # Replace any nans in the 'ethnicity' col with 'Unknown'
    df_resampled['ethnicity'].fillna('Unknown', inplace=True)

    # Initialize a counter for synthetic IDs
    synthetic_id_counter = defaultdict(int)

    # Assign patient IDs to resampled data
    new_patient_ids = []
    last_gene = None
    count = 0
    for key in df_resampled['unique_key']:
        if key in original_patient_id_mapping:
            # If this key is in the original data, use the associated patient ID
            new_patient_ids.append(original_patient_id_mapping[key])
        else:
            # Otherwise, create a new synthetic ID
            gene = key.split('_')[-1]
            if last_gene is None:
                last_gene = gene
                count = -1
            if last_gene == gene:
                count += 1
                if count > 1:
                    count = 0
            else:
                count = 0
            synthetic_id_counter[key] += 1
            synthetic_id = f'synthetic_{gene}_{count}'
            new_patient_ids.append(synthetic_id)
            last_gene = gene

    df_resampled['patient_id_family_id'] = new_patient_ids
    # Drop the unique key as it is no longer needed
    df_resampled.drop('unique_key', axis=1, inplace=True)

    return df_resampled, le_gene, label_encoder_ethnicity


def assign_genes_to_clusters(cluster_counts, clustering_data):
    # Initialize a dictionary to store the gene assigned to each cluster
    assigned_genes = {}

    # Create a dictionary to store the counts for each cluster
    cluster_totals = defaultdict(float)
    for (cluster, gene), count in cluster_counts.items():
        cluster_totals[cluster] += count

    # Initialize a set to store the genes that have been assigned
    assigned_gene_set = set()
    # While there are still clusters left and we haven't assigned 23 genes yet
    while cluster_counts and len(assigned_genes) < 23:
        # Find the cluster with the highest count for its most prevalent gene
        max_gene_cluster = None
        max_gene = 'None'
        for cluster in cluster_totals.keys():
            if cluster_totals[cluster] == 0:  # Skip clusters that don't contain any genes
                continue
            cluster_genes = [(count / cluster_totals[cluster], gene) for (c, gene), count in cluster_counts.items() if c == cluster and gene not in assigned_gene_set]
            if cluster_genes:
                current_max_gene = max(cluster_genes)[1]
                if max_gene_cluster is None or current_max_gene > max_gene:
                    max_gene_cluster = cluster
                    max_gene = current_max_gene

        # If max_gene is 'None', it means there are no more genes to assign
        if max_gene == 'None':
            break

        # Assign this gene to the cluster
        assigned_genes[max_gene_cluster] = max_gene

        # Add this gene to the set of assigned genes
        assigned_gene_set.add(max_gene)

        # Remove this gene from all clusters
        cluster_counts = {(c, gene): count for (c, gene), count in cluster_counts.items() if gene != max_gene}

        # Remove the cluster from the dictionary
        del cluster_totals[max_gene_cluster]

    # Filter the clustering_data DataFrame to only include rows where the cluster and gene match the assigned genes
    clustering_data = clustering_data[clustering_data.apply(lambda row: row['gene'] == assigned_genes.get(row['cluster']), axis=1)]

    return assigned_genes, cluster_counts, clustering_data


def count_genes_in_clusters(data):
    # Create a count of each gene in each cluster
    cluster_counts = defaultdict(int)
    for cluster, gene in zip(data['cluster'], data['gene']):
        cluster_counts[(cluster, gene)] += 1
    return cluster_counts

def cross_validate_clusters(clustering_features, df, n_clusters=23, n_splits=3):
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits)

    matches = 0
    mismatches = 0

    # For each fold, train on the remaining folds and test on the current fold
    for train_index, test_index in skf.split(clustering_features, df['gene']):
        # Split the data into a training set and a test set
        train_features = clustering_features[train_index]
        test_features = clustering_features[test_index]

        # Train the KMeans model on the training data
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto', max_iter=1000)
        kmeans_model.fit(train_features)

        # Create the clustering data for the training set
        train_data = pd.DataFrame({
            'x': train_features[:, 0],
            'y': train_features[:, 1],
            'z': train_features[:, 2],
            'cluster': kmeans_model.labels_,
            'gene': df.iloc[train_index]['gene'].values
        })

        # Use the training data to create the clusters and assign the genes
        cluster_counts_train = count_genes_in_clusters(train_data)
        assigned_genes, _, _ = assign_genes_to_clusters(cluster_counts_train, train_data)

        # For each data point in the test set, assign it to the nearest cluster
        test_clusters = kmeans_model.predict(test_features)

        # Create the clustering data for the test set
        test_data = pd.DataFrame({
            'x': test_features[:, 0],
            'y': test_features[:, 1],
            'z': test_features[:, 2],
            'cluster': test_clusters,
            'gene': df.iloc[test_index]['gene'].values
        })

        # Check if the gene of the data point matches the gene assigned to the cluster
        for data_point, cluster in zip(test_data.itertuples(), test_clusters):
            if data_point.gene == assigned_genes.get(cluster):
                matches += 1
            else:
                mismatches += 1

        print(f"Matches: {matches}")
        print(f"Mismatches: {mismatches}")
        print(f"Accuracy: {matches / (matches + mismatches)}")


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

    # remove outliers
    df = remove_outliers_zscore_per_age_group_and_gene(df)

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

    # Apply K-means before dynamic class balancing
    kmeans_before = KMeans(n_clusters=23,n_init='auto', max_iter=1000)  # Adjust the number of clusters as needed
    processed_data['cluster_before'] = kmeans_before.fit_predict(processed_data[features])

    # Calculate gene percentages in each cluster before dynamic class balancing
    cluster_counts_before = processed_data.groupby('cluster_before')['gene'].value_counts(normalize=True)

   # Get top 3 genes for each cluster before dynamic class balancing
    top_genes_before = []
    for cluster in sorted(processed_data['cluster_before'].unique()):
        top_genes = cluster_counts_before[cluster].nlargest(3).to_dict()
        top_genes_before.append({'cluster': int(cluster), 'top3': top_genes})

    # Use dynamic class balancing to balance the dataset
    df, label_encoder_gene, label_encoder_ethnicity = dynamic_class_balancing(df, features, target, label_encoder_gene, label_encoder_ethnicity)

    # Apply K-means after dynamic class balancing
    kmeans_after = KMeans(n_clusters=23, n_init='auto', max_iter=1000)  # Adjust the number of clusters as needed
    df['cluster_after'] = kmeans_after.fit_predict(df[features])

    # Calculate gene percentages in each cluster after dynamic class balancing
    cluster_counts_after = df.groupby('cluster_after')['gene'].value_counts(normalize=True)

    # Get top 3 genes for each cluster after dynamic class balancing
    top_genes_after = []
    for cluster in sorted(df['cluster_after'].unique()):
        top_genes = cluster_counts_after[cluster].nlargest(23).to_dict()
        top_genes_after.append({'cluster': int(cluster), 'top3': top_genes})

   # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=3)
    clustering_features = tsne.fit_transform(df[features])

    clustering_labels = kmeans_after.fit_predict(clustering_features)

    # Include gene information in the DataFrame from the start
    clustering_data = pd.DataFrame({
    'x': clustering_features[:, 0],
    'y': clustering_features[:, 1],
    'z': clustering_features[:, 2],
    'cluster': clustering_labels,
    'gene': df['gene'].values
    })

    # Convert the Series to a dictionary for easier manipulation
    cluster_counts_dict = cluster_counts_after.to_dict()

    cluster_counts_dict_copy = cluster_counts_dict.copy()

    clustering_data_copy = clustering_data.copy()

    # Create a count of each gene in each cluster
    cluster_counts_dict = defaultdict(int)
    for cluster, gene in zip(clustering_data_copy['cluster'], clustering_data_copy['gene']):
        cluster_counts_dict[(cluster, gene)] += 1

    # Assign one gene to each cluster
    assigned_genes, cluster_counts_after, clustering_data = assign_genes_to_clusters(cluster_counts_dict, clustering_data)

    # see which genes are not present after assigning genes to clusters
    genes_not_present = set(GENES) - set(assigned_genes.values())

    print("Genes not present: ", genes_not_present)

    # Calculate gene percentages in each cluster after applying the autoencoder and K-means clustering
    cluster_counts_after = clustering_data.groupby('cluster')['gene'].value_counts(normalize=True)

    cross_validate_clusters(clustering_features, df, n_clusters=23, n_splits=3)

    # Get top 3 genes for each cluster after applying the autoencoder and K-means clustering
    top_genes_after_encoding = []
    for cluster in sorted(clustering_data['cluster'].unique()):
        # Get counts for this cluster
        cluster_counts = {gene: count for (cluster_key, gene), count in cluster_counts_dict.items() if cluster_key == cluster}
        # Convert to a Series for nlargest
        cluster_counts_series = pd.Series(cluster_counts)
        top_genes = cluster_counts_series.nlargest(3).to_dict()
        top_genes_after_encoding.append({'cluster': int(cluster), 'top3': top_genes})

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

    # Get a count of each gene
    gene_counts = processed_data['gene'].value_counts()

    # Sort the counts from lowest to highest
    gene_counts = gene_counts.sort_values()

    # Convert the Series to a dictionary
    gene_counts_dict = gene_counts.to_dict()

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
        'topGenesBefore': top_genes_before,
        'topGenesAfter': top_genes_after,
        'top_genes_after_encoding': top_genes_after_encoding,
        'geneCounts': gene_counts_dict
    }

    print("Completed data visualization")

    return jsonify(visualization_data)

@app.route('/predict/audiogenev4', methods=['POST'])
def predict_audiogenev4():
    try:
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
        filepath = os.path.join(shared_path, file.filename)
        try:
            file.save(filepath)
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': f'Error saving file: {e}'}), 500

        base_service = '/audiogene-web-service-'
        file_path = os.path.splitext(filepath)[0]

        # Get convert the input file to a csv
        csv_file_path = f"{file_path}.csv"
        print("csv file path: ", csv_file_path)
        container = get_container_by_name(docker_client, base_service + 'xlstocsv-1')
        cmd = f'java -jar ConvertToXLS2CSV.jar {filepath} {csv_file_path}'
        # convert xls to csv using python
        # import pandas as pd
        # data_xls = pd.read_excel(filepath, index_col=None)
        # data_xls.to_csv(csv_file_path, encoding='utf-8', index=False)
        container.exec_run(cmd)

        # Apply perl preprocessing
        post_processing_path = f"{file_path}_processed.out"
        container = get_container_by_name(docker_client, base_service + 'perl_preprocessor-1')
        cmd = f'perl weka-preprocessor.perl -i -a -poly=3 {csv_file_path} > {post_processing_path}'
        res = container.exec_run(cmd).output
        with open(post_processing_path, 'wb') as f:
            f.write(res)

        # Send processed file to the classifier
        predictions_path = f"{file_path}_predictions.csv"
        container = get_container_by_name(docker_client, base_service + 'audiogenev4-1')
        cmd = f'java -jar -Xmx2G AudioGene.jar {post_processing_path} audiogene.misvm.model > {predictions_path}'
        res = container.exec_run(cmd).output

        # Save predictions to a file
        with open(predictions_path, 'wb') as f:
            f.write(res.replace(b'\t', b','))  # Replace the sporadic tab seperation with comma seperation

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
        data.columns = [x for x in range(1, data.shape[1] + 1)]
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
    except Exception as e:
        print(e)
        return jsonify({'error': f'Error predicting{e}'}), 500


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
        # filepath = os.path.join(SHARED_DIR, file.filename)
        filepath = os.path.join(shared_path, file.filename)
        # check if shared dir exists if not create it
        if not os.path.exists(shared_path):
            os.makedirs(shared_path)
        try:
            file.save(filepath)
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': f'Error saving file to share_dir: {e}'}), 404

        base_service = '/audiogene-web-service-'
        file_path = os.path.splitext(filepath)[0]

        o = os.path.join(shared_path, 'ag9_output.csv')
        # Save the output to a temporary location
        # Create an empty csv file
        open(o, 'a').close()

        m = './audiogene_ml/audiogene-ml/notebooks/saved_models/cur_model_lv3_compression.joblib'
        i = filepath

        cmd = f'python -u ./audiogene_ml/audiogene-ml/predict.py -i {i} -o {o} -m {m}'
        print("running command: ", cmd)
        container = get_container_by_name(docker_client, base_service + 'audiogenev9-1')
        container.exec_run(cmd)

        print("finished running command")

        # Read the result using pandas
        data = pd.read_csv(o, index_col=0)

        print(data)

        outputFormat = request.form.get('format', 'json')
        # Return the result in the desired format
        if outputFormat == 'json':
            print("returning json")
            return jsonify(data.to_dict(orient='index')), 200
        elif outputFormat == 'csv':
            print("returning csv")
            csv_data = data.to_csv(index=False)
            return csv_data, 200, {'Content-Type': 'text/csv'}
        else:
            print("returning error")
            return jsonify({'error': 'Unsupported output format'}), 400
    except Exception as e:
        print(e)
        return jsonify({'error': f'Error predicting{e}'}), 500


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
    # app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    app.run(host='0.0.0.0', port=os.getenv('API_INTERNAL_PORT', 5001), debug=True)

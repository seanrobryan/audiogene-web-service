import logging
import os
import subprocess
from collections import defaultdict
import docker
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

app = Flask(__name__)

# docker_client = docker.from_env()
SHARED_DIR = '/' + 'shared_data'

try:
    docker_client = docker.from_env()
    docker_client.containers.list()
except Exception as e:
    print(f"Error connecting to Docker: {e}")
    docker_client = None

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

# function that creates random sample 
def random_sampling(df, n):
    # Ensure at least one sample from each class
    df_sampled = df.groupby('gene').apply(lambda x: x.sample(1)).reset_index(drop=True)
    remaining_samples = n - len(df_sampled)

    if remaining_samples > 0:
        # Randomly sample the remaining data points
        remaining_df = df.loc[~df.index.isin(df_sampled.index)]
        additional_samples = remaining_df.sample(n=remaining_samples, replace=False)
        df_sampled = pd.concat([df_sampled, additional_samples]).reset_index(drop=True)

    X_res = df_sampled.drop('gene', axis=1)
    y_res = df_sampled['gene']
    return X_res, y_res

def dynamic_class_balancing(df, feature_columns, target_column, label_encoder_gene, label_encoder_ethnicity, class_balancing_methods, totalSamples, underrepresented_threshold=40, overrepresented_threshold=250):
    # Define the target and features
    print("class balancing methods: ", class_balancing_methods)
    target = df['gene']
    feature_columns = feature_columns + ['ethnicity_encoded']
    features = df[feature_columns]
    # Create a unique key for each original sample by summing up the frequency labels and appending the gene
    df['unique_key'] = df[feature_columns].sum(axis=1).astype(str) + '_' + df['gene'].astype(str)
    original_keys = df['unique_key'].values
    original_patient_id_mapping = df.set_index('unique_key')['patient_id_family_id'].to_dict()

    # Encode target labels
    le_gene = label_encoder_gene
    y_encoded = le_gene.fit_transform(target)

    # Encode feature labels 'ethnicity'
    label_encoder_ethnicity = label_encoder_ethnicity

    # Define dynamic sampling strategy for SMOTE and NearMiss
    class_counts = np.bincount(y_encoded)
    smote_strategy = {i: underrepresented_threshold for i, count in enumerate(class_counts) if count < underrepresented_threshold}
    nm_strategy = {i: overrepresented_threshold for i, count in enumerate(class_counts) if count > overrepresented_threshold}

    # Apply SMOTE for oversampling and NearMiss for undersampling
    if "SMOTE" in class_balancing_methods and "NearMiss" in class_balancing_methods:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=2)
        nm = NearMiss(sampling_strategy=nm_strategy)
        X_res, y_res = smote.fit_resample(features, y_encoded)
        X_res, y_res = nm.fit_resample(X_res, y_res)
    elif "SMOTE" in class_balancing_methods:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=2)
        X_res, y_res = smote.fit_resample(features, y_encoded)
    elif "NearMiss" in class_balancing_methods:
        nm = NearMiss(sampling_strategy=nm_strategy)
        X_res, y_res = nm.fit_resample(features, y_encoded)
    else:
        features_with_gene = features.copy()
        features_with_gene['gene'] = y_encoded
        X_res, y_res = random_sampling(features_with_gene, n=totalSamples)

    # Reconstruct DataFrame
    df_resampled = pd.DataFrame(X_res, columns=features.columns)
    df_resampled['gene'] = le_gene.inverse_transform(y_res)
    df_resampled['gene_encoded'] = y_res
    df_resampled['ethnicity'] = label_encoder_ethnicity.inverse_transform(df_resampled['ethnicity_encoded'])
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
    print("Resampled data completed")
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
            cluster_genes = sorted([(count / cluster_totals[cluster], gene) for (c, gene), count in cluster_counts.items() if c == cluster and gene not in assigned_gene_set], reverse=True)
            if cluster_genes:
                current_max_gene = cluster_genes[0][1]  # Get the most prevalent gene
                if max_gene_cluster is None or current_max_gene > max_gene:
                    max_gene_cluster = cluster
                    max_gene = current_max_gene

        # If max_gene is 'None', it means there are no more genes to assign to this cluster
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

    # Return whether all genes have been assigned to a cluster
    all_genes_assigned = len(assigned_genes) == 23

    return assigned_genes, cluster_counts, clustering_data, all_genes_assigned

def assign_genes_to_clusters_greedy(cluster_counts, clustering_data, clustering_features, processed_data):
    assigned_genes, cluster_counts, clustering_data, all_genes_assigned = assign_genes_to_clusters(cluster_counts, clustering_data)
    i = 0
    # If not all genes have been assigned to a cluster, recluster and try again
    while not all_genes_assigned:
        # Recluster
        kmeans_after = KMeans(n_clusters=23, n_init='auto', max_iter=1000)  # Adjust the number of clusters as needed
        clustering_labels = kmeans_after.fit_predict(clustering_features)

        clustering_data = pd.DataFrame({
            'x': clustering_features[:, 0],
            'y': clustering_features[:, 1],
            'z': clustering_features[:, 2],
            'cluster': clustering_labels,
            'gene': processed_data['gene'].values
        })

        # Create a count of each gene in each cluster
        cluster_counts_dict = count_genes_in_clusters(clustering_data)

        # Try to assign one gene to each cluster again
        assigned_genes, cluster_counts, clustering_data, all_genes_assigned = assign_genes_to_clusters(cluster_counts_dict, clustering_data)

        # If we have tried to recluster 10 times, give up
        i += 1
        if i == 10:
            break

    return assigned_genes, cluster_counts, clustering_data, all_genes_assigned
    
def count_genes_in_clusters(data):
    # Create a count of each gene in each cluster
    cluster_counts = defaultdict(int)
    for cluster, gene in zip(data['cluster'], data['gene']):
        cluster_counts[(cluster, gene)] += 1
    return cluster_counts


def apply_dbscan_and_convert_to_dict(features, processed_data):
    # Adjust the eps and min_samples parameters to better suit your dataset
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # These values are just examples
    clustering_labels_dbscan = dbscan.fit_predict(features)
    clustering_data_dbscan = pd.DataFrame({
        'x': features[:, 0],
        'y': features[:, 1],
        'z': features[:, 2],
        'cluster': clustering_labels_dbscan,
        'gene': processed_data['gene'].values
    })
    return clustering_data_dbscan.to_dict(orient='records')


def process_additional_data(processed_data, features, gene_labels):
    # Heatmap data
    heatmap = processed_data[features + ['gene_encoded', 'ethnicity_encoded']]
    heatmap = heatmap.rename(columns={'gene_encoded': 'gene', 'ethnicity_encoded': 'ethnicity'})
    heatmap = heatmap.corr()
    heatmap_data = heatmap.to_dict(orient='index')
    
    # Scatter data
    scatter_data = processed_data[['ethnicity', 'gene']].copy()
    scatter_data = scatter_data.rename(columns={'ethnicity': 'x', 'gene': 'y'})
    scatter_data = scatter_data.to_dict(orient='records')

    # Gene Expression data
    gene_expression_data = processed_data[['age', 'gene_encoded']].copy()
    gene_expression_data['gene'] = gene_expression_data['gene_encoded'].apply(lambda x: gene_labels[x])
    gene_expression_data = gene_expression_data.to_dict(orient='records')

    # Audiograms Per Gene
    audiograms_per_gene = processed_data['gene'].value_counts().to_dict()

    # Hearing Loss Over Age
    hearing_loss_over_age = processed_data.groupby('age', as_index=False)[features].mean().to_dict(orient='records')

    # Hearing Loss By Frequency
    features_without_age = [f for f in features if f != 'age']
    hearing_loss_by_frequency = {feature: processed_data[feature].mean() for feature in features_without_age}

    # Hearing Loss Distribution
    hearing_loss_distribution = processed_data[features].values.flatten()
    hearing_loss_distribution = pd.Series(hearing_loss_distribution).value_counts().to_dict()

    # Ethnicity Distribution
    ethnicity_distribution = processed_data['ethnicity'].value_counts().to_dict()

    return heatmap_data, scatter_data, gene_expression_data, audiograms_per_gene, hearing_loss_over_age, hearing_loss_by_frequency, hearing_loss_distribution, ethnicity_distribution


def calculate_top_genes_and_gene_counts(processed_data, kmeans, cluster_col_before, cluster_col_after, processed_data_before):
    # # Top Genes Before
    # cluster_counts_before = processed_data.groupby(cluster_col_before)['gene'].value_counts(normalize=True)
    # top_genes_before = [{int(cluster): cluster_counts_before[cluster].nlargest(3).to_dict()} for cluster in sorted(processed_data[cluster_col_before].unique())]
    top_genes_before = []

    # # Top Genes After
    # cluster_counts_after = processed_data.groupby(cluster_col_after)['gene'].value_counts(normalize=True)
    # top_genes_after = [{int(cluster): cluster_counts_after[cluster].nlargest(3).to_dict()} for cluster in sorted(processed_data[cluster_col_after].unique())]
    top_genes_after = []

    # Gene Counts
    gene_counts = processed_data_before['gene'].value_counts().sort_values().to_dict()

    gene_counts_after_resampling = processed_data['gene'].value_counts().sort_values().to_dict()

    return top_genes_before, top_genes_after, gene_counts, gene_counts_after_resampling


def save_to_excel(processed_data, clustering_data, path):
    csv_file_path = os.path.join(path, 'all_data.xlsx')
    with pd.ExcelWriter(csv_file_path) as writer:
        processed_data.to_excel(writer, sheet_name='Data')
        clustering_data.to_excel(writer, sheet_name='Clustering Data')
        # Add more sheets as required

def prepare_table_df_data(table_df, table_df_features, table_df_labels, closest_genes, closest_clusters, distances):
    table_df_data = pd.DataFrame({
        'id': table_df['id'].values,
        'x': table_df_features[:, 0],
        'y': table_df_features[:, 1],
        'z': table_df_features[:, 2],
        'cluster': table_df_labels,
        'closest_genes': [closest_genes[i:i+3] for i in range(0, len(closest_genes), 3)],
        'closest_clusters': [closest_clusters[i:i+3] for i in range(0, len(closest_clusters), 3)],
        'distances_to_closest_genes': [distances[i] for i in range(len(distances))]
    })
    table_df_data = table_df_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return table_df_data.to_dict(orient='records')


@app.route('/data_visualization', methods=['POST'])
def data_visualization():
    # Get the data from the request
    data = request.get_json()

    class_balancing_methods = data['classBalancingMethod']
    minThreshold = data['minThreshold']
    maxThreshold = data['maxThreshold']
    totalSamples = data['totalSamples']
    print("class balancing: ", class_balancing_methods)
    print("min threshold: ", minThreshold)
    print("max threshold: ", maxThreshold)

    print("The data: ", data)
    print("Starting data visualization")

    # Convert JSON data to DataFrame and drop 'ear' column
    table_df = pd.json_normalize(data['data']).drop(['ear'], axis=1)

    # Load main dataset
    df = pd.read_csv(CSV_FILE_PATH)

    # Rename columns using a mapping dictionary
    column_mapping = {'{} dB'.format(i): '{} Hz'.format(i) for i in [125, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]}
    df.rename(columns=column_mapping, inplace=True)
    table_df.rename(columns=column_mapping, inplace=True)

    # Remove outliers and preprocess data
    df = remove_outliers_zscore_per_age_group_and_gene(df)
    features = list(column_mapping.values()) + ['age']
    target = 'gene'
    processed_data = df[features + [target, 'ethnicity', 'patient_id_family_id', 'family_id']].copy()

    # Impute missing values
    processed_data.fillna(processed_data.mean(numeric_only=True), inplace=True)
    processed_data.fillna('NA', inplace=True)
    frequency_columns = [col for col in processed_data.columns if 'Hz' in col]
    for column in frequency_columns:
        if column in table_df.columns:
            # Get rows within +/- 5 years of each row's age
            for i, row in table_df.iterrows():
                age_range = range(row['age'] - 5, row['age'] + 6)
                similar_age_rows = processed_data[processed_data['age'].isin(age_range)]
                # If the value is missing in the table_df DataFrame, fill it with the mean value from the similar_age_rows DataFrame
                if pd.isnull(table_df.loc[i, column]):
                    table_df.loc[i, column] = similar_age_rows[column].mean()
            # After filling the missing values for the current column, interpolate the remaining missing values in the column
            table_df[column] = table_df[column].interpolate(method='linear', limit_direction='both')

    # Encoding and clustering
    label_encoder_gene, label_encoder_ethnicity = LabelEncoder(), LabelEncoder()
    processed_data['gene_encoded'] = label_encoder_gene.fit_transform(processed_data[target])
    processed_data['ethnicity_encoded'] = label_encoder_ethnicity.fit_transform(processed_data['ethnicity'])
    gene_labels = label_encoder_gene.classes_  # Save the original gene labels

    kmeans = KMeans(n_clusters=23, n_init='auto', max_iter=1000)  # Adjust the number of clusters as needed
    processed_data['cluster_before'] = kmeans.fit_predict(processed_data[features])

    processed_data_before = processed_data.copy()

    # Dynamic class balancing and clustering
    processed_data, label_encoder_gene, label_encoder_ethnicity = dynamic_class_balancing(processed_data, features, target, label_encoder_gene, label_encoder_ethnicity, class_balancing_methods, totalSamples, underrepresented_threshold=minThreshold, overrepresented_threshold=maxThreshold)
    processed_data['cluster_after'] = kmeans.fit_predict(processed_data[features])

    # Dimensionality reduction
    pca = PCA(n_components=3)
    clustering_features = pca.fit_transform(processed_data[features])
    clustering_labels = kmeans.fit_predict(clustering_features)
    table_df_features = pca.transform(table_df[features])
    table_df_labels = kmeans.predict(table_df_features)

    # KNN for closest genes
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(clustering_features)
    distances, indices = knn.kneighbors(table_df_features)
    closest_genes = processed_data.iloc[indices.flatten()]['gene'].values
    closest_clusters = clustering_labels[indices.flatten()]

    # Data preparation for visualization
    clustering_data = pd.DataFrame({
        'x': clustering_features[:, 0],
        'y': clustering_features[:, 1],
        'z': clustering_features[:, 2],
        'cluster': clustering_labels,
        'gene': processed_data['gene'].values
    })

    # Prepare table_df_data with closest genes and clusters
    table_df_data = prepare_table_df_data(table_df, table_df_features, table_df_labels, closest_genes, closest_clusters, distances)

    # Convert table_df to a dictionary
    table_df = table_df.to_dict(orient='records')

    
    # Clustering data states
    clustering_data_original = clustering_data.to_dict(orient='records')  # Original clustering data
    cluster_counts_dict = count_genes_in_clusters(clustering_data)  # Count of each gene in each cluster
    assigned_genes, cluster_counts, clustering_data_assigned_genes, all_genes_assigned = assign_genes_to_clusters_greedy(cluster_counts_dict, clustering_data, clustering_features, processed_data)  # Greedy gene assignment
    clustering_data_assigned_genes = clustering_data_assigned_genes.to_dict(orient='records')  # Greedy clustering data
    clustering_data_dbscan = apply_dbscan_and_convert_to_dict(clustering_features, processed_data)  # DBSCAN clustering data

    print("Finished clustering")

    # Other data processing and conversion to dictionaries
    heatmap_data, scatter_data, gene_expression_data, audiograms_per_gene, hearing_loss_over_age, hearing_loss_by_frequency, hearing_loss_distribution, ethnicity_distribution = process_additional_data(processed_data, features, gene_labels)

    # Calculate top genes and gene counts
    top_genes_before, top_genes_after, gene_counts_dict, gene_counts_after_resampling = calculate_top_genes_and_gene_counts(processed_data, kmeans, 'cluster_before', 'cluster_after', processed_data_before)

    # Saving data to Excel
    save_to_excel(processed_data, clustering_data, path=os.path.dirname(os.path.abspath(__file__)))

    print("Finished data processing")

    # Combine all data into a single JSON object
    visualization_data = {
        'heatmapData': heatmap_data,
        'clusteringDataGreedy': clustering_data_assigned_genes,
        'clusteringDataOriginal': clustering_data_original,
        'clusteringDataOptimized': clustering_data_dbscan,
        'ageData': processed_data['age'].to_dict(),
        'scatterData': scatter_data,
        'geneExpressionData': gene_expression_data,
        'audiogramsPerGene': audiograms_per_gene,
        'hearingLossOverAge': hearing_loss_over_age,
        'hearingLossByFrequency': hearing_loss_by_frequency,
        'hearingLossDistribution': hearing_loss_distribution,
        'ethnicityDistribution': ethnicity_distribution,
        'topGenesBefore': top_genes_before,
        'topGenesAfter': top_genes_after,
        'geneCounts': gene_counts_dict,
        'geneCountsAfterResampling': gene_counts_after_resampling,
        'table_df_data': table_df_data,
        'table_df': table_df,
    }

    print("Finished data visualization", visualization_data)

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
    # For testing
    # app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    # For use with Docker
    app.run(host='0.0.0.0', port=os.getenv('API_INTERNAL_PORT', 5001), debug=True, use_reloader=False)

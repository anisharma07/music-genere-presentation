"""
Flask Web Application for Music Genre Classification
Supports KMeans and Spectral Clustering with different train-test splits
"""

from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
from feature_extraction import extract_features, features_to_array
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load scaler and label encoder
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Cache for loaded models
models_cache = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(algorithm, split):
    """Load model from cache or disk"""
    model_key = f"{algorithm}_{split}"

    if model_key not in models_cache:
        split_formatted = split.replace('-', '_')
        model_path = f"models/{algorithm}_{split_formatted}.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            models_cache[model_key] = pickle.load(f)

    return models_cache[model_key]


def predict_with_kmeans(model, features):
    """Predict genre using KMeans model"""
    cluster = model.predict(features)[0]

    # Get distances to all cluster centers
    distances = np.linalg.norm(model.cluster_centers_ - features, axis=1)

    # Get confidence score (inverse of distance to assigned cluster)
    confidence = 1 / (1 + distances[cluster])

    # Get top 3 predictions
    sorted_indices = np.argsort(distances)[:3]
    top_predictions = []

    for idx in sorted_indices:
        genre = label_encoder.classes_[idx]
        dist = distances[idx]
        conf = 1 / (1 + dist)
        top_predictions.append({
            'genre': genre,
            'confidence': float(conf * 100)
        })

    return cluster, confidence, top_predictions


def predict_with_spectral(model_data, features):
    """Predict genre using Spectral Clustering model"""
    # For spectral clustering, we use nearest neighbor approach
    # since it doesn't have a traditional predict method
    X_train = model_data['X_train']
    train_labels = model_data['train_labels']

    # Find k nearest neighbors in training data
    k = 10
    similarities = cosine_similarity(features, X_train)[0]
    nearest_indices = np.argsort(similarities)[-k:]

    # Get the most common cluster among nearest neighbors
    nearest_clusters = train_labels[nearest_indices]
    cluster = np.bincount(nearest_clusters).argmax()

    # Calculate confidence based on similarity to nearest neighbors
    confidence = np.mean(similarities[nearest_indices])

    # Get top 3 predictions based on cluster frequencies
    unique_clusters, counts = np.unique(nearest_clusters, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1][:3]

    top_predictions = []
    total_count = len(nearest_clusters)

    for idx in sorted_indices:
        cluster_id = unique_clusters[idx]
        genre = label_encoder.classes_[cluster_id]
        conf = (counts[idx] / total_count) * 100
        top_predictions.append({
            'genre': genre,
            'confidence': float(conf)
        })

    return cluster, confidence, top_predictions


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models and splits"""
    return jsonify({
        'algorithms': ['kmeans', 'spectral'],
        'splits': ['50-50', '60-40', '70-30', '80-20'],
        'genres': label_encoder.classes_.tolist()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict genre from uploaded audio file"""
    try:
        # Check if file is present
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio_file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Get algorithm and split
        algorithm = request.form.get('algorithm', 'kmeans')
        split = request.form.get('split', '80-20')

        if algorithm not in ['kmeans', 'spectral']:
            return jsonify({'error': 'Invalid algorithm'}), 400

        if split not in ['50-50', '60-40', '70-30', '80-20']:
            return jsonify({'error': 'Invalid split'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Extract features
            print(f"Extracting features from {filename}...")
            features_dict = extract_features(filepath)
            features_array = features_to_array(features_dict)

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Load model and predict
            print(f"Loading {algorithm} model with {split} split...")
            model = load_model(algorithm, split)

            if algorithm == 'kmeans':
                cluster, confidence, top_predictions = predict_with_kmeans(
                    model, features_scaled)
            else:  # spectral
                cluster, confidence, top_predictions = predict_with_spectral(
                    model, features_scaled)

            # Get primary prediction
            predicted_genre = top_predictions[0]['genre']

            response = {
                'success': True,
                'predicted_genre': predicted_genre,
                'confidence': float(confidence * 100),
                'cluster_id': int(cluster),
                'top_predictions': top_predictions,
                'algorithm': algorithm,
                'split': split,
                'filename': filename
            }

            print(
                f"Prediction: {predicted_genre} ({confidence*100:.2f}% confidence)")
            return jsonify(response)

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models_cache),
        'genres': label_encoder.classes_.tolist()
    })


if __name__ == '__main__':
    print("="*80)
    print("Music Genre Classification Server")
    print("="*80)
    print(f"Available genres: {', '.join(label_encoder.classes_)}")
    print(f"Algorithms: KMeans, Spectral Clustering")
    print(f"Train-Test Splits: 50-50, 60-40, 70-30, 80-20")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print("="*80)
    print("Starting server on http://localhost:5000")
    print("="*80)

    app.run(debug=True, host='0.0.0.0', port=5000)

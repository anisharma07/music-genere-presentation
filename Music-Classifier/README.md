# Music Genre Classifier

A machine learning web application for classifying music genres using KMeans and Spectral Clustering algorithms.

## Features

- 🎵 **Multiple Clustering Algorithms**: KMeans and Spectral Clustering
- 📊 **Various Train-Test Splits**: 50-50, 60-40, 70-30, 80-20
- 🎼 **Audio Feature Extraction**: Extracts 58 audio features from uploaded files
- 🌐 **Web Interface**: Beautiful and intuitive Flask-based web UI
- 📁 **Multiple Format Support**: WAV, MP3, FLAC, OGG, M4A

## Dataset

The application uses the GTZAN Genre Collection dataset with the following features:
- Chroma STFT
- RMS Energy
- Spectral Centroid, Bandwidth, and Rolloff
- Zero Crossing Rate
- Harmony and Perceptr
- Tempo
- MFCCs (20 coefficients)

## Project Structure

```
Music-Classifier/
├── gtzan.csv                  # Dataset with extracted features
├── train_models.py            # Script to train clustering models
├── feature_extraction.py      # Audio feature extraction module
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Web interface
├── models/                    # Trained models (generated after training)
│   ├── kmeans_50_50.pkl
│   ├── kmeans_60_40.pkl
│   ├── kmeans_70_30.pkl
│   ├── kmeans_80_20.pkl
│   ├── spectral_50_50.pkl
│   ├── spectral_60_40.pkl
│   ├── spectral_70_30.pkl
│   ├── spectral_80_20.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── uploads/                   # Temporary folder for uploaded files
```

## Installation

1. **Clone the repository**:
```bash
cd "Desktop/Final Github/Music-Classifier"
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Models

First, train the clustering models on the GTZAN dataset:

```bash
python train_models.py
```

This will:
- Load the GTZAN dataset from `gtzan.csv`
- Train KMeans and Spectral Clustering models for each train-test split
- Save all models in the `models/` directory
- Display evaluation metrics (ARI, NMI, Silhouette Score)

Expected output:
```
Training models for different train-test splits...
[KMeans - 50-50] - Train ARI: X.XXXX, NMI: X.XXXX, Silhouette: X.XXXX
[Spectral - 50-50] - Train ARI: X.XXXX, NMI: X.XXXX, Silhouette: X.XXXX
...
Models saved successfully!
```

### Step 2: Start the Flask Server

Run the web application:

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Step 3: Use the Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Select the clustering algorithm (KMeans or Spectral)
3. Choose the train-test split ratio
4. Upload an audio file (first 30 seconds will be analyzed)
5. Click "Predict Genre"
6. View the prediction results with confidence scores

## API Endpoints

### GET `/api/models`
Get available models and genres.

**Response**:
```json
{
  "algorithms": ["kmeans", "spectral"],
  "splits": ["50-50", "60-40", "70-30", "80-20"],
  "genres": ["blues", "classical", "country", ...]
}
```

### POST `/api/predict`
Predict genre from uploaded audio file.

**Parameters**:
- `audio_file`: Audio file (multipart/form-data)
- `algorithm`: "kmeans" or "spectral"
- `split`: "50-50", "60-40", "70-30", or "80-20"

**Response**:
```json
{
  "success": true,
  "predicted_genre": "blues",
  "confidence": 85.32,
  "cluster_id": 0,
  "top_predictions": [
    {"genre": "blues", "confidence": 85.32},
    {"genre": "jazz", "confidence": 72.14},
    {"genre": "rock", "confidence": 65.89}
  ],
  "algorithm": "kmeans",
  "split": "80-20",
  "filename": "song.wav"
}
```

### GET `/api/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": 4,
  "genres": ["blues", "classical", "country", ...]
}
```

## Features Extracted

The system extracts 58 audio features from each audio file:

1. **Temporal Features**: Length
2. **Spectral Features**: 
   - Chroma STFT (mean, variance)
   - Spectral Centroid (mean, variance)
   - Spectral Bandwidth (mean, variance)
   - Spectral Rolloff (mean, variance)
3. **Energy Features**: RMS Energy (mean, variance)
4. **Time-Domain Features**: Zero Crossing Rate (mean, variance)
5. **Harmonic Features**: Harmony and Perceptr (mean, variance)
6. **Rhythmic Features**: Tempo
7. **Cepstral Features**: 20 MFCCs (mean, variance for each)

## Model Performance

The models are evaluated using:
- **Adjusted Rand Index (ARI)**: Measures clustering agreement with true labels
- **Normalized Mutual Information (NMI)**: Measures shared information between clusters and labels
- **Silhouette Score**: Measures cluster cohesion and separation

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (KMeans, Spectral Clustering)
- **Audio Processing**: librosa (feature extraction)
- **Data Processing**: pandas, numpy
- **Frontend**: HTML, CSS, JavaScript

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'librosa'"
**Solution**: Install all dependencies with `pip install -r requirements.txt`

### Issue: "FileNotFoundError: Model not found"
**Solution**: Run `python train_models.py` to train and save the models first

### Issue: "Error extracting features"
**Solution**: Ensure the audio file is in a supported format (WAV, MP3, FLAC, OGG, M4A)

### Issue: Server not starting
**Solution**: Make sure port 5000 is not already in use, or modify the port in `app.py`

## Notes

- Only the first 30 seconds of uploaded audio files are analyzed
- Maximum file size: 50MB
- Models are cached in memory after first load for faster predictions
- Uploaded files are automatically deleted after processing

## Future Enhancements

- [ ] Add supervised classification models (SVM, Random Forest, Neural Networks)
- [ ] Support for real-time audio streaming
- [ ] Visualization of audio features
- [ ] Model comparison dashboard
- [ ] Export predictions to CSV
- [ ] Batch processing for multiple files

## License

This project is for educational purposes.

## Contact

For questions or issues, please create an issue in the repository.

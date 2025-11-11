# Quick Start Guide - Music Genre Classifier

## 🚀 Quick Start

### 1. Installation (Already Complete!)
✅ Python environment configured
✅ All packages installed
✅ Models trained and ready
✅ Flask server running

### 2. Access the Application

Open your web browser and navigate to:
**http://localhost:5000**

Or alternatively:
- http://127.0.0.1:5000
- http://192.168.138.61:5000

### 3. Using the Web Interface

#### Step 1: Select Algorithm
Choose between:
- **KMeans Clustering** (Faster, partition-based)
- **Spectral Clustering** (Graph-based, better for complex patterns)

#### Step 2: Choose Train-Test Split
Select the model trained on your preferred split:
- **80-20** (Recommended - more training data)
- 70-30
- 60-40
- 50-50

#### Step 3: Upload Audio File
- Click the file upload area
- Select an audio file (WAV, MP3, FLAC, OGG, M4A)
- Only the first 30 seconds will be analyzed
- Maximum file size: 50MB

#### Step 4: Predict Genre
- Click "Predict Genre" button
- Wait for the analysis (takes 5-15 seconds depending on file size)
- View the results:
  - Primary predicted genre
  - Confidence score
  - Top 3 predictions with probabilities
  - Cluster ID
  - Algorithm and split used

### 4. Example Usage Flow

```
1. Open http://localhost:5000
2. Select "KMeans Clustering"
3. Select "80-20" split
4. Upload a music file (e.g., blues_song.wav)
5. Click "Predict Genre"
6. View results: "BLUES" with 85% confidence
```

## 📊 Model Performance

Based on the training results:

### KMeans Clustering
| Split | Train ARI | Test ARI | Train NMI | Test NMI |
|-------|-----------|----------|-----------|----------|
| 50-50 | 0.1806    | 0.1637   | 0.3325    | 0.3207   |
| 60-40 | 0.1871    | 0.1757   | 0.3404    | 0.3290   |
| 70-30 | 0.1689    | 0.1609   | 0.3192    | 0.3402   |
| 80-20 | 0.1763    | 0.1516   | 0.3153    | 0.3479   |

### Spectral Clustering
| Split | Train ARI | Train NMI |
|-------|-----------|-----------|
| 50-50 | 0.1655    | 0.3631    |
| 60-40 | 0.1580    | 0.3489    |
| 70-30 | 0.1367    | 0.3445    |
| 80-20 | 0.1398    | 0.3341    |

**Metrics Explained:**
- **ARI (Adjusted Rand Index)**: Measures clustering agreement (0-1, higher is better)
- **NMI (Normalized Mutual Information)**: Measures information sharing (0-1, higher is better)
- **Silhouette Score**: Measures cluster quality (−1 to 1, higher is better)

## 🎵 Supported Genres

The model can classify music into 10 genres:
1. Blues
2. Classical
3. Country
4. Disco
5. Hip Hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

## 🔍 How It Works

### Feature Extraction (58 features)
When you upload an audio file, the system extracts:

1. **Temporal Features** (1)
   - Length

2. **Spectral Features** (8)
   - Chroma STFT (mean, variance)
   - Spectral Centroid (mean, variance)
   - Spectral Bandwidth (mean, variance)
   - Spectral Rolloff (mean, variance)

3. **Energy Features** (2)
   - RMS Energy (mean, variance)

4. **Time-Domain Features** (2)
   - Zero Crossing Rate (mean, variance)

5. **Harmonic Features** (4)
   - Harmony (mean, variance)
   - Perceptr (mean, variance)

6. **Rhythmic Features** (1)
   - Tempo

7. **Cepstral Features** (40)
   - 20 MFCCs × 2 (mean, variance)

### Clustering Process

1. **Feature Standardization**: Features are scaled using StandardScaler
2. **Clustering**: KMeans or Spectral Clustering groups similar music
3. **Prediction**: New audio is matched to the nearest cluster
4. **Genre Mapping**: Cluster is mapped to the most likely genre

## 🛠️ API Endpoints

### GET /api/models
Returns available models and genres.

**Example:**
```bash
curl http://localhost:5000/api/models
```

### POST /api/predict
Predicts genre from uploaded audio.

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "audio_file=@song.wav" \
  -F "algorithm=kmeans" \
  -F "split=80-20"
```

### GET /api/health
Health check endpoint.

**Example:**
```bash
curl http://localhost:5000/api/health
```

## 📝 Common Issues & Solutions

### Server Not Starting
**Issue**: Port 5000 already in use
**Solution**: Change port in `app.py` (line 234):
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Upload Fails
**Issue**: File too large
**Solution**: Ensure file is under 50MB or increase limit in `app.py` (line 14)

### Feature Extraction Error
**Issue**: Unsupported audio format
**Solution**: Convert to WAV, MP3, FLAC, OGG, or M4A

### Low Accuracy
**Issue**: Clustering may not always be accurate
**Solution**: 
- Try different algorithm (KMeans vs Spectral)
- Use 80-20 split for best results
- Consider that unsupervised clustering has limitations compared to supervised learning

## 🔄 Retraining Models

To retrain models with different parameters:

```bash
# Edit train_models.py to modify:
# - Number of clusters
# - Clustering parameters
# - Train-test splits

# Then run:
python train_models.py
```

## 🌐 Deployment

### For Production Use:

1. **Use a production WSGI server:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Use environment variables for configuration:**
```python
import os
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))
```

3. **Add authentication if needed**

4. **Use HTTPS in production**

## 📈 Improving Accuracy

Current approach uses unsupervised clustering. To improve:

1. **Use Supervised Learning:**
   - Train Random Forest, SVM, or Neural Networks
   - Requires labeled training data
   - Expected accuracy: 70-85%

2. **Increase Features:**
   - Add more audio features
   - Use deep learning embeddings
   - Apply dimensionality reduction (PCA, t-SNE)

3. **Ensemble Methods:**
   - Combine multiple models
   - Voting or stacking

4. **Data Augmentation:**
   - Pitch shifting
   - Time stretching
   - Add noise

## 🎯 Testing the Application

### Test with Sample Audio

1. Find or download sample audio files in different genres
2. Upload to the web interface
3. Compare predicted genre with actual genre
4. Try different algorithms and splits

### Expected Results

- Blues songs should cluster with similar patterns (slow tempo, specific chord progressions)
- Classical music has distinct harmonic features
- Metal/Rock have higher energy and spectral characteristics
- Hip Hop has distinct rhythmic patterns

## 📚 Additional Resources

### Audio Feature Explanation
- **Chroma STFT**: Represents pitch classes (C, C#, D, etc.)
- **Spectral Centroid**: "Brightness" of sound
- **MFCCs**: Represent timbre/texture of sound
- **Tempo**: Beats per minute

### Learning More
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Music Information Retrieval](https://musicinformationretrieval.com/)

## 🐛 Debugging

Enable verbose logging in `app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs for:
- Feature extraction errors
- Model loading issues
- Prediction failures

## 📊 Project Statistics

- **Total Code Files**: 4 Python scripts + 1 HTML
- **Models Trained**: 8 (4 KMeans + 4 Spectral)
- **Features Extracted**: 58 per audio file
- **Supported Formats**: 5 audio formats
- **Genres Classified**: 10 music genres
- **Train-Test Splits**: 4 configurations

## 🎉 Congratulations!

Your Music Genre Classifier is now fully operational!

Happy classifying! 🎵🎸🎹🎤

---

**Server Status**: ✅ Running on http://localhost:5000
**Models**: ✅ Loaded and ready
**Ready to classify**: ✅ Yes!

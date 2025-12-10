FMA: A Dataset For Music Analysis
Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.
International Society for Music Information Retrieval Conference (ISMIR), 2017.

We introduce the Free Music Archive (FMA), an open and easily accessible dataset suitable for evaluating several tasks in MIR, a field concerned with browsing, searching, and organizing large music collections. The community's growing interest in feature and end-to-end learning is however restrained by the limited availability of large audio datasets. The FMA aims to overcome this hurdle by providing 917 GiB and 343 days of Creative Commons-licensed audio from 106,574 tracks from 16,341 artists and 14,854 albums, arranged in a hierarchical taxonomy of 161 genres. It provides full-length and high-quality audio, pre-computed features, together with track- and user-level metadata, tags, and free-form text such as biographies. We here describe the dataset and how it was created, propose a train/validation/test split and three subsets, discuss some suitable MIR tasks, and evaluate some baselines for genre recognition. Code, data, and usage examples are available at https://github.com/mdeff/fma.

Paper: arXiv:1612.01840 (latex and reviews)
Slides: doi:10.5281/zenodo.1066119
Poster: doi:10.5281/zenodo.1035847
Data
All metadata and features for all tracks are distributed in fma_metadata.zip (342 MiB). The below tables can be used with pandas or any other data analysis tool. See the paper or the usage.ipynb notebook for a description.

tracks.csv: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
genres.csv: all 163 genres with name and parent (used to infer the genre hierarchy and top-level genres).
features.csv: common features extracted with librosa.
echonest.csv: audio features provided by Echonest (now Spotify) for a subset of 13,129 tracks.
Then, you got various sizes of MP3-encoded audio data:

fma_small.zip: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
fma_medium.zip: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
fma_large.zip: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
fma_full.zip: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)
See the wiki (or #41) for known issues (errata).

Code
The following notebooks, scripts, and modules have been developed for the dataset.

usage.ipynb: shows how to load the datasets and develop, train, and test your own models with it.
analysis.ipynb: exploration of the metadata, data, and features. Creates the figures used in the paper.
baselines.ipynb: baseline models for genre recognition, both from audio and features.
features.py: features extraction from the audio (used to create features.csv).
webapi.ipynb: query the web API of the FMA. Can be used to update the dataset.
creation.ipynb: creation of the dataset (used to create tracks.csv and genres.csv).
creation.py: creation of the dataset (long-running data collection and processing).
utils.py: helper functions and classes.
Usage
Binder   Click the binder badge to play with the code and data from your browser without installing anything.

Clone the repository.

git clone https://github.com/mdeff/fma.git
cd fma
Create a Python 3.6 environment.
Install dependencies.

pip install --upgrade pip setuptools wheel
pip install numpy==1.12.1  # workaround resampy's bogus setup.py
pip install -r requirements.txt
Note: you may need to install ffmpeg or graphviz depending on your usage.
Note: install CUDA to train neural networks on GPUs (see Tensorflow's instructions).

Download some data, verify its integrity, and uncompress the archives.

cd data

curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_large.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_full.zip

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip"   | sha1sum -c -
echo "497109f4dd721066b5ce5e5f250ec604dc78939e  fma_large.zip"    | sha1sum -c -
echo "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  fma_full.zip"     | sha1sum -c -

unzip fma_metadata.zip
unzip fma_small.zip
unzip fma_medium.zip
unzip fma_large.zip
unzip fma_full.zip

cd ..
Note: try 7zip if decompression errors. It might be an unsupported compression issue.

Fill a .env configuration file (at repository's root) with the following content.

AUDIO_DIR=./data/fma_small/  # the path to a decompressed fma_*.zip
FMA_KEY=MYKEY  # only if you want to query the freemusicarchive.org API
Open Jupyter or run a notebook.

jupyter notebook
make usage.ipynb
Impact, coverage, and resources
100+ research papers
2 derived works
~10 posts
5 events
~10 dataset lists
Contributing
Contribute by opening an issue or a pull request. Let this repository be a hub around the dataset!

History
2017-05-09 pre-publication release

paper: arXiv:1612.01840v2
code: git tag rc1
fma_metadata.zip sha1: f0df49ffe5f2a6008d7dc83c6915b31835dfe733
fma_small.zip sha1: ade154f733639d52e35e32f5593efe5be76c6d70
fma_medium.zip sha1: c67b69ea232021025fca9231fc1c7c1a063ab50b
fma_large.zip sha1: 497109f4dd721066b5ce5e5f250ec604dc78939e
fma_full.zip sha1: 0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab
known issues: see #41
2016-12-06 beta release

paper: arXiv:1612.01840v1
code: git tag beta
fma_small.zip sha1: e731a5d56a5625f7b7f770923ee32922374e2cbf
fma_medium.zip sha1: fe23d6f2a400821ed1271ded6bcd530b7a8ea551
Acknowledgments and Licenses
We are grateful to the Swiss Data Science Center (EPFL and ETHZ) for hosting the dataset.

Please cite our work if you use our code or data.

@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
@inproceedings{fma_challenge,
  title = {Learning to Recognize Musical Genre from Audio},
  subtitle = {Challenge Overview},
  author = {Defferrard, Micha\"el and Mohanty, Sharada P. and Carroll, Sean F. and Salath\'e, Marcel},
  booktitle = {The 2018 Web Conference Companion},
  year = {2018},
  publisher = {ACM Press},
  isbn = {9781450356404},
  doi = {10.1145/3184558.3192310},
  archiveprefix = {arXiv},
  eprint = {1803.05337},
  url = {https://arxiv.org/abs/1803.05337},
}
The code in this repository is released under the MIT license.
The metadata is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
We do not hold the copyright on the audio and distribute it under the license chosen by the artist.
The dataset is meant for research purposes.

Carl Thomé · Updated 6 years ago

Code

Download
GTZAN Genre Collection
Musical audio files with corresponding single-label genre tags.

About Dataset
Context
This dataset was used for the well known paper in genre classification "Musical genre classification of audio signals" by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

Unfortunately the database was collected gradually and very early on in my research so I have no titles (and obviously no copyright permission etc). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. Nevetheless I have been providing it to researchers upon request mainly for comparison purposes etc. Please contact George Tzanetakis (gtzan@cs.uvic.ca) if you intend to publish experimental results using this dataset.

There are some practical and conceptual issues with this dataset, described in "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use" by B. Sturm on arXiv 2013.

The lesson is not to banish GTZAN, but to use it with consideration of its contents.

I personally find GTZAN to be a good starting point for learning more about deep learning (DL) in music information retrieval (MIR) and why it's hard sometimes because the ground truth is often subjective, context-sensitive and not categorically true.

Content
The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050 Hz monophonic 16-bit audio files in .au format.


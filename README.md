## Quickstart Guide  
  
### If using included sample songs:  
1. Clone repo
2. Run main.py
3. View results (30_sec_data.csv, 30_sec_performance.csv 30_sec_holdout.csv --> these will be created after main.py is run)  
  
### If using own songs:  
1. Delete songs in sample_music_files
2. Put songs to be used in sample_music_files (**Note:** If you change the name of this folder, make sure to also change it at top of main.py; same goes for changing the name of folder sample_music_files_sliced)
3. Run main.py
4. View results (30_sec_data.csv, 30_sec_performance.csv 30_sec_holdout.csv --> these will be created after main.py is run)  
  
**Note:** Names of files containing results will reflect parameter of clip_sec in main.py and may not match the file names in the 'View results' steps above  
  
# Is it Major or Minor? Classifying the Mode of a Song  
### By Alex Smith  
  
**Overview: Problem and Background**  
  
Going into this project, I knew I wanted to work with audio. After doing some research, I settled on working with music by directly extracting features from mp3s. I chose to work with music data because it is a type of audio that can evoke emotion in addition to thought. When I listen to music, I sometimes ask myself: Why does a particular song make me feel happy or sad? The key of a song helps determine the feeling of a song. There are two parts that make up the key, the tonic note (also known as the root note or base note), and the model (either major or minor). Going in to this project, I posed the question: Can I predict the mode?
  
**Data**  
  
To collect my data, I downloaded more than 1,000 mp3s in the form of Billboard Top 100 collections. These varied by year, and represent what music is most popular at a given time. This step gave me raw audio files. From there I dove into Digital Signal Processing (DSP) and Music Information Retrieval (MIR). Using the Python-compatible libraries Librosa and Pydub, I sliced all of the mp3s into clips of the first 30 seconds of the song and extracted several audio features. I settled on the first 30 seconds after comparing model performance at 10 seconds, 20 seconds, 25 seconds, and 30 seconds.
  
The final features I was able to extract were: Title (str), BPM (int), Tonic note (str, int), Zero-crossing-rate mean (int), Zero-crossing-rate std (int), Spectral-centroid mean (int), Spectral-centroid std (int), Root-mean-square energy mean (int), Root-mean-square energy std (int), and my target variable: Mode (str, int).
  
The balance between the two classes of my target variable, Mode, were 59% Minor and 41% Major.
  
**Project Design**  
  
One of my personal goals for this project was to create .py files in PyCharm that automated the process of extracting data and building and analyzing models. I am happy to say that I accomplished this goal and divided my project code into four files: extract.py, transform.py, model.py, and main.py, in addition to having these steps in Jupyter Notebook with visualizations.
  
**Tools**  
  
The main tools that I used in addition to Python, PyCharm, and Jupyter Notebook were Pandas, Librosa, Pydub, and Scikit-Learn. I decided to use Librosa and Pydub for working with raw audio files after doing several hours of research into the different tools available in the fields of DSP and MIR. Librosa has a wide variety of functions that allow for an in-depth analysis of audio files, which allowed me to become familiar with the specific challenges and fascinating parts of working with audio signal processing.

I directly used a function in my code to estimate the key from a pitch class distribution, written by bmcfee (Brian McFee).  Thank you for the awesome work Brian! https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd
  
**Algorithm / Results**  
  
In total, I built, trained, tested, and evaluated 13 classification models. The models include: (1) Logistic Regression Baseline, (2) Logistic Regression Optimized with Grid Search, (3) K-Nearest Neighbor Baseline, (4) K-Nearest Neighbor Optimized with Grid Search, (5) Support Vector Classifier Baseline, (6) Support Vector Classifier Baseline (Normalized), (7) Support Vector Classifier Optimized with Grid Search (Normalized), (8) Support Vector Classifier Optimized with Random Search (Normalized), (9) Naive Bayes Classifier, (10) Decision Tree Baseline, (11) Decision Tree Optimized with Grid Search, (12) Random Forest Baseline, and (13) Random Forest Optimized with Random Search.
  
The performance metric that I optimized for was accuracy, as the problem I am addressing does not have high costs for misclassification. The three models with the highest accuracy scores can be seen below:
  
<img width="422" alt="screen shot 2018-05-16 at 5 50 17 pm" src="https://user-images.githubusercontent.com/34464435/40150973-d18edf96-5931-11e8-8f88-4783569b5af4.png">
  
I tested the performance of these three models with my holdout data and these were the final results:  
  
<img width="228" alt="screen shot 2018-05-16 at 5 50 29 pm" src="https://user-images.githubusercontent.com/34464435/40150988-f29807e4-5931-11e8-9ef7-2219273f7926.png">
  
Though my Optimized Decision Tree model and my optimized Random Forest model performed equally well, the higher levels of interpretability and simplicity of the Decision Tree model let me to choosing this as by best model. Comparing the AUC scores between all my models confirmed my previous findings that Optimized Decision Tree and Optimized Random Forest were the best performing models.
  
The visualization of my best model: Decision Tree Optimized with Grid Search.  
  
![decision tree](https://user-images.githubusercontent.com/34464435/40151007-1a069ffc-5932-11e8-9373-145c6001c952.png)
  
**Future Direction**  
  
Moving forward, I would love to incorporate more mp3s into my dataset to train my model(s). As I have my code setup in PyCharm to automate the entire process from feature extraction to pre-processing to model training, test, and evaluation, it would be incredibly easy to add more data. Additionally, I would like to implement a Deep Learning model to answer my stated question as I think there can be a higher performing model built to accurately classify a song as Major or Minor. I am aware that Deep Learning models are often used for music, audio, and signal processing, so that is another reason in favor of building a Deep Learning model.
  
**Challenges**  
  
1. Accuracy of Mode label (Major or Minor) in data - I found the accuracy of the automated labeling technique I was using to be about 70% based on a random sample (using other sources to confirm or reject the label or Major or Minor).  
  
2. Setting up a Config file - This is something I really wanted to implement in my PyCharm code so that anyone could easily implement my model. However, it was outside of the scope of the project for me as I was not able to address it fully within the time frame.
  
**Contact**  
  
email: datagrlxyz @ gmail dot com  
twitter: @datagrlxyz  
blog: www.datagrl.xyz

Datasets can be download from here:
https://drive.google.com/file/d/0B5OyMUIx8-rHbEo4OHpaQWhOelE/view?usp=sharing


I conducted all of my experiments using Weka functions from custom scripts. This script was constructed with in Groovy,
a Java implementation that simplifies some of the coding process.

Further information on groovy and Weka can be found here:
https://weka.wikispaces.com/Using+WEKA+from+Groovy

Two scripts were used:
1. ML_experiment.groovy     - used for experimenting with parameters
2. ML_learning_curve.groovy - used for constructing the final learning curve (% training data vs error)

These scripts were executed in the following pattern:
groovy -classpath weka.jar <path_to_script> <path_to_data_directory> <dataset> <model>

Possible <dataset> values:
1. adult
2. news
3. adult_norm - adult data with normalized attributes (used for kNN experiment)
4. adult_best - adult data with only the most significant attributes (used for kNN experiment)
5. news_norm - adult data with normalized values (used for kNN experiment)
6. news_best - news data with only the most significant attributes (used for kNN experiment)

Possible <model> values:
1. DT - decision trees
2. NN - neural networks
3. ADA - boosting
4. kNN - k-nearest neighbors
5. SVM_1 - SVM with linear kernel
6. SVM_2 - SVM with RBF kernel

For example, if I wanted to run neural network on the news data, I would execute the follow command:

groovy -classpath weka.jar ~/Documents/CS7641/Project1/Scripts/ML_experiment.groovy ~/Documents/CS7641/Project1/Datasets/ news NN > ~/Documents/CS7641/Project1/news/Logs/logfile &
// Learning Algorithms
import weka.classifiers.trees.J48
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.meta.AdaBoostM1
import weka.core.neighboursearch.LinearNNSearch
import weka.classifiers.lazy.IBk
import weka.classifiers.functions.supportVector.RBFKernel
import weka.classifiers.functions.supportVector.PolyKernel
import weka.classifiers.functions.SMO

import weka.filters.unsupervised.instance.RemovePercentage
import weka.filters.Filter

// Data Handling and Evaluation
import weka.core.converters.ConverterUtils.DataSource
import weka.core.Instances
import weka.classifiers.Evaluation

// Load appropriate data based on command-line argument




train_error = []
test_error = []
 
 
String filepath = args[0]


println args[2] + ' on ' + args[1] + ':'

// DECISION TREES

// MinNumObj Experiment
if (args[2] == 'DT'){


    for(int perc=90; perc>-1; perc-=10){

            if (args[1] == 'adult'){

                train_path = filepath + "/adult_train.arff"
                test_path = filepath + "/adult_test.arff"

                m = 7


            } else if (args[1] == 'news'){
                train_path = filepath + "/news_train.arff"
                test_path = filepath + "/news_test.arff"

                m = 381


            }


            // load data and set class index
            train = DataSource.read(train_path)
            train.setClassIndex(train.numAttributes() - 1)
            test = DataSource.read(test_path)
            test.setClassIndex(test.numAttributes() - 1)

            filt = new RemovePercentage()
            String[] options_filt = ["-P",perc]
            filt.setOptions(options_filt)
            filt.setInputFormat(train)
            train = filt.useFilter(train,filt)



        train_error = []
        test_error = []
        elapsedtime = []

        println 100-perc + "% of Total Dataset"



            // Set Options
            String[] options = ["-M",m]

            // create the model
            DT = new J48()
            DT.setOptions(options)

            // Build and Time
            long starttime = System.currentTimeMillis();
            DT.buildClassifier(train)
            long stoptime = System.currentTimeMillis();

            elapsedtime.addAll((stoptime - starttime)/1000);

            // evaluate
            evaluation = new Evaluation(train)

            evaluation.evaluateModel(DT,train)



            train_error.addAll(evaluation.pctIncorrect())

            evaluation = new Evaluation(train)
            evaluation.evaluateModel(DT,test)
            test_error.addAll(evaluation.pctIncorrect())


       // System.out.println(evaluation.toSummaryString())

        // Output Error and Training Time
        println train_error
        println test_error
        println elapsedtime
        println " "

    }


// NEURAL NETWORK
} else if (args[2] == 'NN'){


    for(int perc=90; perc>-1; perc-=10){

        if (args[1] == 'adult'){
                train_path = filepath + "/adult_train.arff"
                test_path = filepath + "/adult_test.arff"

            h = 5
            a = 0.1

        } else if (args[1] == 'news'){
                train_path = filepath + "/news_train.arff"
                test_path = filepath + "/news_test.arff"

            h = 2
            a = 0.1

        }

        n = 500

        // load data and set class index
        train = DataSource.read(train_path)
        train.setClassIndex(train.numAttributes() - 1)
        test = DataSource.read(test_path)
        test.setClassIndex(test.numAttributes() - 1)


        filt = new RemovePercentage()
        String[] options_filt = ["-P",perc]
        filt.setOptions(options_filt)
        filt.setInputFormat(train)
        train = filt.useFilter(train,filt)



        train_error = []
        test_error = []
        elapsedtime = []

        println 100-perc + "% of Total Dataset"

        for(int s=1; s<11; s++){


            // Set Options
            String[] options = ["-S",s,"-H",h,"-L",a,"-N",n]

            // create the model
            NN = new MultilayerPerceptron()
            NN.setOptions(options)

            // Build and Time
            long starttime = System.currentTimeMillis();
            NN.buildClassifier(train)
            long stoptime = System.currentTimeMillis();

            elapsedtime.addAll((stoptime - starttime)/1000);


            // evaluate
            evaluation = new Evaluation(train)
            evaluation.evaluateModel(NN,train)
            // train_error = [train_error,evaluation.pctIncorrect()]
            train_error.addAll(evaluation.pctIncorrect())

            evaluation = new Evaluation(train)
            evaluation.evaluateModel(NN,test)
            test_error.addAll(evaluation.pctIncorrect())
        }

        // println "Training Error %"
        println train_error

        // println "Testing Error %"
        println test_error

        // print Elapsed Time
        println elapsedtime
        println " "


    }


//ADABOOST
} else if (args[2] == 'ADA'){

     for(int perc=90; perc>-1; perc-=10){

        if (args[1] == 'adult'){
                train_path = filepath + "/adult_train.arff"
                test_path = filepath + "/adult_test.arff"

            m = 70


        } else if (args[1] == 'news'){
                train_path = filepath + "/news_train.arff"
                test_path = filepath + "/news_test.arff"

            m = 1


        }


        // load data and set class index
        train = DataSource.read(train_path)
        train.setClassIndex(train.numAttributes() - 1)
        test = DataSource.read(test_path)
        test.setClassIndex(test.numAttributes() - 1)


        filt = new RemovePercentage()
        String[] options_filt = ["-P",perc]
        filt.setOptions(options_filt)
        filt.setInputFormat(train)
        train = filt.useFilter(train,filt)

        println 100-perc + "% of Total Dataset"

        //Set Options
        String[] options = ["-P", "100", "-S", "1", "-I", 50, "-W","weka.classifiers.trees.J48","--", "-C", "0.25", "-M", m]



        // create the model
        ADA = new AdaBoostM1()
        ADA.setOptions(options)

        // Build and Time
        long starttime = System.currentTimeMillis();
        ADA.buildClassifier(train)
        long stoptime = System.currentTimeMillis();

        elapsedtime = ((stoptime - starttime)/1000);

        // evaluate
        evaluation = new Evaluation(train)
        evaluation.evaluateModel(ADA,train)
        // train_error = [train_error,evaluation.pctIncorrect()]
        train_error = evaluation.pctIncorrect()

        evaluation = new Evaluation(train)
        evaluation.evaluateModel(ADA,test)
        test_error = evaluation.pctIncorrect()

        // println "Training Error %"
        println train_error

        // println "Testing Error %"
        println test_error

        // print Elapsed Time
        println elapsedtime
        println " "

    }




} else if (args[2] == "kNN"){


     for(int perc=90; perc>-1; perc-=10){

        if (args[1] == 'adult'){
                train_path = filepath + "/adult_train.arff"
                test_path = filepath + "/adult_test.arff"

            k = 26


        } else if (args[1] == 'news'){
                train_path = filepath + "/news_train.arff"
                test_path = filepath + "/news_test.arff"

            k = 29


        }


            // load data and set class index
        train = DataSource.read(train_path)
        train.setClassIndex(train.numAttributes() - 1)
        test = DataSource.read(test_path)
        test.setClassIndex(test.numAttributes() - 1)

        filt = new RemovePercentage()
        String[] options_filt = ["-P",perc]
        filt.setOptions(options_filt)
        filt.setInputFormat(train)
        train = filt.useFilter(train,filt)

        println 100-perc + "% of Total Dataset"

            //Set Options
            String[] options = ["-K",k]

            // create the model
            kNN = new IBk()
            kNN.setOptions(options)

            // Build and Time

            kNN.buildClassifier(train)


            // evaluate
            evaluation = new Evaluation(train)
            evaluation.evaluateModel(kNN,train)

            evaluation = new Evaluation(train)
            long starttime = System.currentTimeMillis();
            evaluation.evaluateModel(kNN,test)
            long stoptime = System.currentTimeMillis();
            elapsedtime = ((stoptime - starttime)/1000);

            test_error = evaluation.pctIncorrect()
            // println "Training Error %"
            println "[0]"

            // println "Testing Error %"
            println test_error

            // print Elapsed Time
            println elapsedtime
            println " "
    }

    //Polykernal
} else if (args[2] == "SVM") {


    for(int perc=90; perc>-1; perc-=10){

    if (args[1] == 'adult'){
                train_path = filepath + "/adult_train.arff"
                test_path = filepath + "/adult_test.arff"

        method = "RBFKernel"
        c = 10

    } else if (args[1] == 'news'){
                train_path = filepath + "/news_train.arff"
                test_path = filepath + "/news_test.arff"

        method = "RBFKernel"
        c = 10

    }


    // load data and set class index
    train = DataSource.read(train_path)
    train.setClassIndex(train.numAttributes() - 1)
    test = DataSource.read(test_path)
    test.setClassIndex(test.numAttributes() - 1)

        filt = new RemovePercentage()
        String[] options_filt = ["-P",perc]
        filt.setOptions(options_filt)
        filt.setInputFormat(train)
        train = filt.useFilter(train,filt)

        println 100-perc + "% of Total Dataset"



    confs = [0.1,1,10,100]
    //confs = [100]



        String[] options = ["-C",c,"-L",0.001,"-N",0,"-V",1,"-K","weka.classifiers.functions.supportVector." + method]

        // create the model
        SVM = new SMO()
        SVM.setOptions(options)

        // Build and Time
        long starttime = System.currentTimeMillis();
        SVM.buildClassifier(train)
        long stoptime = System.currentTimeMillis();

        elapsedtime = ((stoptime - starttime)/1000);

        // evaluate
        evaluation = new Evaluation(train)
        evaluation.evaluateModel(SVM,train)
        // train_error = [train_error,evaluation.pctIncorrect()]
        train_error = evaluation.pctIncorrect()

        evaluation = new Evaluation(train)
        evaluation.evaluateModel(SVM,test)
        test_error = evaluation.pctIncorrect()

        // println "Training Error %"
        println train_error

        // println "Testing Error %"
        println test_error

        // print Elapsed Time
        println elapsedtime
        println " "

}


    }


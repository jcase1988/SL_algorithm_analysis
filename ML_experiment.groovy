// Learning Algorithms
import weka.classifiers.trees.J48
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.meta.AdaBoostM1
import weka.core.neighboursearch.LinearNNSearch
import weka.classifiers.lazy.IBk
import weka.classifiers.functions.supportVector.RBFKernel
import weka.classifiers.functions.supportVector.PolyKernel
import weka.classifiers.functions.SMO



// Data Handling and Evaluation
import weka.core.converters.ConverterUtils.DataSource
import weka.core.Instances
import weka.classifiers.Evaluation

String filepath = args[0]


// Load appropriate data based on command-line argument
if (args[1] == 'adult'){
    train_path = filepath + "/adult_train.arff"
    test_path = filepath + "/adult_CV.arff"

} else if (args[1] == 'news'){
    train_path = filepath + "/News_train.arff"
    test_path = filepath + "/News_CV.arff"

} else if (args[1] == 'adult_best'){
    train_path = filepath + "/adult_train_best_Attributes.arff"
    test_path = filepath + "/adult_CV_best_Attributes.arff"

} else if (args[1] == 'news_best'){
    train_path = filepath + "/News_train_best_Attributes.arff"
    test_path = filepath + "/News_CV_best_Attributes.arff"

} else if (args[1] == 'adult_norm'){
    train_path = filepath + "/adult_train_normalized.arff"
    test_path = filepath + "/adult_CV_normalized.arff"

} else if (args[1] == 'news_norm'){
    train_path = filepath + "/News_train_normalized.arff"
    test_path = filepath + "/News_CV_normalized.arff"
}

train_error = []
test_error = []
 
 
// load data and set class index
train = DataSource.read(train_path)
train.setClassIndex(train.numAttributes() - 1)
test = DataSource.read(test_path)
test.setClassIndex(test.numAttributes() - 1)

println args[2] + ' on ' + args[1] + ':'

// DECISION TREES

// MinNumObj Experiment
if (args[2] == 'DT'){



    for(int m=1; m < 101; m++){


        train_error = []
        test_error = []
        elapsedtime = []

        println " -M " + m

        // Random Seeds
        //for(int s=1; s<11; s++){

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

        //}

       // System.out.println(evaluation.toSummaryString())

        // Output Error and Training Time
        println train_error
        println test_error
        println elapsedtime
        println " "

    }


// NEURAL NETWORK

} else if (args[2] == 'NN'){

    learning_rates = [0.3, 0.1]

    for(int ai=0; ai<2; ai++){

        a = learning_rates[ai]

        for(int h=1; h<11; h++){

            train_error = []
            test_error = []
            elapsedtime = []

            println "-H " + h

            for(int s=1; s<11; s++){


                // Set Options
                String[] options = ["-S",s,"-H",h,"-L",a]

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

    }


//ADABOOST
} else if (args[2] == 'ADA'){

    iter = [1, 50, 100, 200]
    minObjNums = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


    for(ii = 0; ii < 7; ii++){

        i = iter[ii]

        for(mi = 0; mi < 11; mi++){

            m = minObjNums[mi]


            println "-i " + i + " -M " + m

            //Set Options
            String[] options = ["-P", "100", "-S", "1", "-I", i, "-W","weka.classifiers.trees.J48","--", "-C", "0.25", "-M", m]



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


    }

} else if (args[2] == "kNN"){

    for(k = 1; k < 101; k++){

            println "-K " + k

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
} else if (args[2] == "SVM_1") {



    confs = [0.1,1,10,100]
    //confs = [100]

    for (ie = 0; ie < 5; ie++){

        c = confs[ie]
        println "-C " + c

        String[] options = ["-C",c,"-L",0.001,"-N",0,"-V",1,"-K","weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1"]

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

} else if (args[2] == "SVM_2") {



    confs = [0.1,1,10,100]

    for (ig = 0; ig < 5; ig++){

        c = confs[ig]

        println "-C " + c
        String[] options = ["-C",c,"-L",0.001,"-N",0,"-V",1,"-K","weka.classifiers.functions.supportVector.RBFKernel"]

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
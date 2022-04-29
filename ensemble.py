import argparse

# Parse any command line arguments
argumentParser = argparse.ArgumentParser()

argumentSchemeA = "a"
argumentSchemeB = "b"
argumentParser.add_argument("-s", "--scheme",
                            dest="scheme",
                            type=str.lower,
                            choices=[argumentSchemeA, argumentSchemeB],
                            default=[argumentSchemeA, argumentSchemeB],
                            nargs="+",
                            metavar="Scheme",
                            help="the scheme - \"a\" (FEBRL), \"b\" (ePBRN), or both - to evaluate.")

argumentImplementationReference = "reference"
argumentImplementationStudent = "student"
argumentParser.add_argument("-i", "--implementation",
                            dest="implementation",
                            type=str.lower,
                            choices=[argumentImplementationReference, argumentImplementationStudent],
                            default=[argumentImplementationReference, argumentImplementationStudent],
                            nargs="+",
                            metavar="Implementation",
                            help="the implementation - \"reference\" (authors'), \"student\" (CS598 project team members'), or both - to execute.")

argumentParser.add_argument("-t", "--train",
                            dest="train",
                            action="store_true",
                            default=False,
                            help="train the base learner models rather than use the pre-trained models.")

argumentParser.add_argument("-d", "--data",
                            dest="data",
                            action="store_true",
                            default=False,
                            help="reprocess the training and test datasets rather than using preprocessed datasets (this may be a time-consuming process).")

argumentParser.add_argument("-e", "--search",
                            dest="search",
                            action="store_true",
                            default=False,
                            help="perform the hyperparameter grid search (this is a VERY time-consuming process!).")

argumentParser.add_argument("-v", "--verbose",
                            dest="verbose",
                            action="store_true",
                            default=False,
                            help="print verbose statistics (as in the original authors' implementation).")

args = argumentParser.parse_args()

# Import the general numeric and data manipulation libraries we'll be using
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import KFold
import copy
import StatusPrinter

# Status/informational message formatting
status = StatusPrinter.StatusPrinter()

# Import the scheme modules specified by the user
if(argumentSchemeA in args.scheme):
    import FEBRL

# Load the datasets, either as pre-processed data or from scratch, as specified by the user
a_x_train = None
a_y_train = None
a_x_test = None
a_y_test = None

a_x_train_tensor = None
a_y_train_tensor = None
a_x_test_tensor = None
a_y_test_tensor = None

b_x_train = None
b_y_train = None
b_x_test = None
b_y_test = None

b_x_train_tensor = None
b_y_train_tensor = None
b_x_test_tensor = None
b_y_test_tensor = None

status.Print("===== DATASET LOADING AND PROCESSING =====")

if(args.data == True):
    # Reprocess the data sets from scratch
    print("STATUS: Processing source datasets (this may take a few minutes)... ", end="")

    if(argumentSchemeA in args.scheme):
        # The following dataset construction routine is taken from the original authors' implementation
        # Construct the training and test data sets for Scheme A from file
        a_trainset = "febrl3_UNSW"
        a_testset = "febrl4_UNSW"

        ## TRAIN SET CONSTRUCTION
        # Import the training data set from file
        print("Import train set...")
        a_df_train = pd.read_csv(a_trainset+".csv", index_col = "rec_id")
        a_train_true_links = FEBRL.generate_true_links(a_df_train)
        print("Train set size:", len(a_df_train), ", number of matched pairs: ", str(len(a_train_true_links)))

        # Convert the postcode field to a string and append phonetically normalized given name/surname fields
        a_df_train['postcode'] = a_df_train['postcode'].astype(str)
        a_df_train['given_name_soundex'] = FEBRL.phonetic(a_df_train['given_name'], method='soundex')
        a_df_train['given_name_nysiis'] = FEBRL.phonetic(a_df_train['given_name'], method='nysiis')
        a_df_train['surname_soundex'] = FEBRL.phonetic(a_df_train['surname'], method='soundex')
        a_df_train['surname_nysiis'] = FEBRL.phonetic(a_df_train['surname'], method='nysiis')

        # Generate the training feature vector matrix and corresponding labels
        a_x_train, a_y_train = FEBRL.generate_train_X_y(a_df_train, a_train_true_links)

        ## TEST SET CONSTRUCTION
        # Import the test data set from file
        print("Import test set...")
        a_df_test = pd.read_csv(a_testset+".csv", index_col = "rec_id")
        a_test_true_links = FEBRL.generate_true_links(a_df_test)
        a_leng_test_true_links = len(a_test_true_links)
        print("Test set size:", len(a_df_test), ", number of matched pairs: ", str(a_leng_test_true_links))

        # Perform blocking on the test dataset to identify candidate pairs
        print("BLOCKING PERFORMANCE:")
        a_blocking_fields = ["given_name", "surname", "postcode"]
        a_all_candidate_pairs = []
        for field in a_blocking_fields:
            block_indexer = FEBRL.rl.BlockIndex(on=field)
            candidates = block_indexer.index(a_df_test)
            a_detects = FEBRL.blocking_performance(candidates, a_test_true_links, a_df_test)
            a_all_candidate_pairs = candidates.union(a_all_candidate_pairs)
            print("Number of pairs of matched "+ field +": "+str(len(candidates)), ", detected ",
                a_detects,'/'+ str(a_leng_test_true_links) + " true matched pairs, missed " + 
                str(a_leng_test_true_links-a_detects) )
        a_detects = FEBRL.blocking_performance(a_all_candidate_pairs, a_test_true_links, a_df_test)
        print("Number of pairs of at least 1 field matched: " + str(len(a_all_candidate_pairs)), ", detected ",
            a_detects,'/'+ str(a_leng_test_true_links) + " true matched pairs, missed " + 
                str(a_leng_test_true_links-a_detects) )

        # Convert the postcode field to a string and append phonetically normalized given name/surname fields
        a_df_test['postcode'] = a_df_test['postcode'].astype(str)
        a_df_test['given_name_soundex'] = FEBRL.phonetic(a_df_test['given_name'], method='soundex')
        a_df_test['given_name_nysiis'] = FEBRL.phonetic(a_df_test['given_name'], method='nysiis')
        a_df_test['surname_soundex'] = FEBRL.phonetic(a_df_test['surname'], method='soundex')
        a_df_test['surname_nysiis'] = FEBRL.phonetic(a_df_test['surname'], method='nysiis')

        # Generate the test feature vector matrix and corresponding labels
        print("Extract feature vectors...")
        a_df_x_test = FEBRL.extract_features(a_df_test, a_all_candidate_pairs)
        a_vectors = a_df_x_test.values.tolist()
        a_labels = [0]*len(a_vectors)
        a_feature_index = a_df_x_test.index
        for i in range(0, len(a_feature_index)):
            if a_df_test.loc[a_feature_index[i][0]]["match_id"]==a_df_test.loc[a_feature_index[i][1]]["match_id"]:
                a_labels[i] = 1
        a_x_test, a_y_test = FEBRL.shuffle(a_vectors, a_labels, random_state=0)
        a_x_test = np.array(a_x_test)
        a_y_test = np.array(a_y_test)
        print("Count labels of y_test:",FEBRL.collections.Counter(a_y_test))
        print("Finished building X_test, y_test")

else:
    # Load the pre-processed datasets
    status.Print("Loading preprocessed training and test datasets... ")

    if(argumentSchemeA in args.scheme):
        # Load the pre-processed datasets for Scheme A
        a_x_train = np.load("pre/a_x_train.npy")
        a_y_train = np.load("pre/a_y_train.npy")
        a_x_test = np.load("pre/a_x_test.npy")
        a_y_test = np.load("pre/a_y_test.npy")

    if(argumentSchemeB in args.scheme):
        # Load the pre-processed datasets for Scheme B
        b_x_train = np.load("pre/b_x_train.npy")
        b_y_train = np.load("pre/b_y_train.npy")
        b_x_test = np.load("pre/b_x_test.npy")
        b_y_test = np.load("pre/b_y_test.npy")

# Convert the numpy arrays into tensors for the student implementation
if((argumentSchemeA in args.scheme) and (argumentImplementationStudent in args.implementation)):
    a_x_train_tensor = torch.from_numpy(a_x_train).float()
    a_y_train_tensor = torch.from_numpy(a_y_train).float()
    a_x_test_tensor = torch.from_numpy(a_x_test).float()
    a_y_test_tensor = torch.from_numpy(a_y_test).long()

#if((argumentSchemeB in args.scheme) and (argumentImplementationStudent in args.implementation)):
    #b_x_train_tensor = torch.from_numpy(b_x_train).float()
    #b_y_train_tensor = torch.from_numpy(b_y_train).float()
    #b_x_test_tensor = torch.from_numpy(b_x_test).float()
    #b_y_test_tensor = torch.from_numpy(b_y_test).long()

print("")

# Evaluate the specified models
if(argumentSchemeA in args.scheme):

    if(argumentImplementationReference in args.implementation):
        # Evaluate the Scheme A dataset against the reference implementation
        print("")
        status.Print("===== MODEL EVALUATION (REFERENCE IMPLEMENTATION) =====")

        a_modeltypes = ['svm', 'nn', 'lg'] 
        a_modeltypes_2 = ['linear', 'relu', 'l2']
        a_modelparams = [0.005, 100, 0.2]
        a_nFold = 10
        a_kf = KFold(n_splits=a_nFold)
        a_model_raw_score = [0]*3
        a_model_binary_score = [0]*3
        a_model_i = 0

        a_base_learners_svm = [None] * a_nFold
        a_base_learners_nn = [None] * a_nFold
        a_base_learners_lg = [None] * a_nFold
        
        if(args.train == True):
            # Train the base learners from scratch
            status.Print("Training reference implementation against Scheme A...")

            status.Indent()
            for a_model_i in range(3):
                
                modeltype = a_modeltypes[a_model_i]
                modeltype_2 = a_modeltypes_2[a_model_i]
                modelparam = a_modelparams[a_model_i]

                message = "Training {} reference implementation base learners...".format(modeltype)
                status.Print(message)

                iFold = 0
                result_fold = [0]*a_nFold
                final_eval_fold = [0]*a_nFold

                base_learner = [None] * a_nFold

                for train_index, valid_index in a_kf.split(a_x_train):
                    X_train_fold = a_x_train[train_index]
                    y_train_fold = a_y_train[train_index]
                    
                    md = FEBRL.train_model(modeltype, modelparam, X_train_fold, y_train_fold, modeltype_2)
                    base_learner[iFold] = md
                    iFold = iFold + 1

                # The following commented lines can be enabled to write this execution's trained models to file
                if(modeltype == 'svm'):
                    a_base_learners_svm = copy.deepcopy(base_learner)
                    #pickle.dump(a_base_learners_svm, open("pre/a_model_reference_svm.pkl", mode='wb'))

                elif(modeltype == 'nn'):
                    a_base_learners_nn = copy.deepcopy(base_learner)
                    #pickle.dump(a_base_learners_nn, open("pre/a_model_reference_nn.pkl", mode='wb'))

                elif(modeltype == 'lg'):
                    a_base_learners_lg = copy.deepcopy(base_learner)
                    #pickle.dump(a_base_learners_lg, open("pre/a_model_reference_lg.pkl", mode='wb'))
            
            status.Unindent()

        else:
            # Load the pre-trained models
            status.Print("Loading pre-trained reference implementation Scheme A...")

            a_base_learners_svm = pickle.load(open("pre/a_model_reference_svm.pkl", mode='rb'))
            a_base_learners_nn = pickle.load(open("pre/a_model_reference_nn.pkl", mode='rb'))
            a_base_learners_lg = pickle.load(open("pre/a_model_reference_lg.pkl", mode='rb'))

        status.Print("Evaluating the reference implementation against Scheme A...")

        # Evaluate the dataset against the base learners
        for a_model_i in range(3):
            modeltype = a_modeltypes[a_model_i]

            message = "Evaluating dataset against {} reference implementation base learners...".format(modeltype)
            status.Print(message)

            base_learner = None

            if(modeltype == 'svm'):
                base_learner = a_base_learners_svm
            elif(modeltype == 'nn'):
                base_learner = a_base_learners_nn
            elif(modeltype == 'lg'):
                base_learner = a_base_learners_lg
            
            result_fold = [0]*a_nFold
            final_eval_fold = [0]*a_nFold

            status.Indent()
            for a_base_learner_i in np.arange(a_nFold):

                result_fold[a_base_learner_i] = FEBRL.classify(base_learner[a_base_learner_i], a_x_test)
                final_eval_fold[a_base_learner_i] = FEBRL.evaluation(a_y_test, result_fold[a_base_learner_i])

                if(args.verbose == True):
                    message = "Base learner {}: {}".format(str(a_base_learner_i), final_eval_fold[a_base_learner_i])
                else:
                    message = "Base learner {}: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(str(a_base_learner_i),
                                final_eval_fold[a_base_learner_i]["precision"],
                                final_eval_fold[a_base_learner_i]["sensitivity"],
                                final_eval_fold[a_base_learner_i]["F-score"])

                status.Print(message, prependTimestamp=False)

            bagging_raw_score = np.average(result_fold, axis=0)
            bagging_binary_score  = np.copy(bagging_raw_score)
            bagging_binary_score[bagging_binary_score > 0.5] = 1
            bagging_binary_score[bagging_binary_score <= 0.5] = 0
            bagging_eval = FEBRL.evaluation(a_y_test, bagging_binary_score)
            
            if(args.verbose == True):
                message = "{} bagging: {}".format(modeltype, bagging_eval)
            else:
                message = "{} bagging: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(modeltype,
                            bagging_eval["precision"],
                            bagging_eval["sensitivity"],
                            bagging_eval["F-score"])

            status.Print(message, prependTimestamp=False)
            print("")

            status.Unindent()

            a_model_raw_score[a_model_i] = bagging_raw_score
            a_model_binary_score[a_model_i] = bagging_binary_score

        # Perform ensemble stacking
        a_thres = .99

        a_stack_raw_score = np.average(a_model_raw_score, axis=0)
        a_stack_binary_score = np.copy(a_stack_raw_score)
        a_stack_binary_score[a_stack_binary_score > a_thres] = 1
        a_stack_binary_score[a_stack_binary_score <= a_thres] = 0
        a_stacking_eval = FEBRL.evaluation(a_y_test, a_stack_binary_score)

        status.Print("Reference Implementation Scheme A stacking performance:")

        if(args.verbose == True):
            message = "{}".format(a_stacking_eval)
        else:
            message = "precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(
                        a_stacking_eval["precision"],
                        a_stacking_eval["sensitivity"],
                        a_stacking_eval["F-score"])

        status.Print(message, prependTimestamp=False)
        print("")

    if(argumentImplementationStudent in args.implementation):
        # Evaluate the Scheme A dataset against the student implementation
        import FEBRLStudentImplementation as fs

        print("")
        status.Print("===== MODEL EVALUATION (STUDENT IMPLEMENTATION) =====")

        a_modeltypes = ['svm', 'nn', 'lg']
        a_base_learner_count = 10
        a_base_learners_svm = [None] * a_base_learner_count
        a_base_learners_nn = [None] * a_base_learner_count
        a_base_learners_lg = [None] * a_base_learner_count

        # For SVM, shift our data from its native range of [0.0, 1.0] to [-1.0, 1.0]
        a_x_train_tensor_svm = (a_x_train_tensor * 2) - 1
        a_y_train_tensor_svm = (a_y_train_tensor * 2) - 1
        a_x_test_tensor_svm = (a_x_test_tensor * 2) - 1
        a_y_test_tensor_svm = (a_y_test_tensor * 2) - 1

        # Selected hyperparameters, determined through grid search
        frs_inverse_reg_optimal = 20  
        frn_weight_decay_optimal = 0.025
        frl_inverse_reg_optimal = 0.5

        if(args.train == True):
            # Train the base learners from scratch
            status.Print("Training student implementation against Scheme A...")

            status.Indent()
            for a_model_i in np.arange(3):
                # Perform bagging across 10 base learners by using shuffled kfold as our sampling strategy for
                # bootstrapping
                modeltype = a_modeltypes[a_model_i]
                status.Print("Training {} student implementation base learners... ".format(modeltype))

                frs_kfold_count = a_base_learner_count
                frs_kfold = KFold(n_splits=frs_kfold_count, shuffle=True, random_state=12345)
                frs_kfold_i = 0

                frs_results = [0] * frs_kfold_count

                for train_indicies, _ in frs_kfold.split(a_x_train):

                    if(modeltype == 'svm'):
                        febrl_reproducer_svm = fs.FEBRLReproducerSVM(num_features=a_x_train_tensor.shape[1], inverse_reg=frs_inverse_reg_optimal)
                        febrl_reproducer_svm.fit(a_x_train_tensor_svm[train_indicies], a_y_train_tensor_svm[train_indicies])
                        a_base_learners_svm[frs_kfold_i] = febrl_reproducer_svm

                    elif(modeltype == 'nn'):
                        febrl_reproducer_nn = fs.FEBRLReproducerNN(num_features=a_x_train_tensor.shape[1], weight_decay=frn_weight_decay_optimal)
                        febrl_reproducer_nn.fit(a_x_train_tensor[train_indicies], a_y_train_tensor[train_indicies])
                        a_base_learners_nn[frs_kfold_i] = febrl_reproducer_nn

                    elif(modeltype == 'lg'):
                        febrl_reproducer_lg = fs.FEBRLReproducerLR(num_features=a_x_train_tensor.shape[1], inverse_reg=frl_inverse_reg_optimal)
                        febrl_reproducer_lg.fit(a_x_train_tensor[train_indicies], a_y_train_tensor[train_indicies])
                        a_base_learners_lg[frs_kfold_i] = febrl_reproducer_lg

                    frs_kfold_i = frs_kfold_i + 1             

                # The following section can be enabled to write this execution's trained models to file
                #if(modeltype == 'svm'):
                    #pickle.dump(a_base_learners_svm, open("pre/a_model_student_svm.pkl", mode='wb'))

                #if(modeltype == 'nn'):
                    #pickle.dump(a_base_learners_nn, open("pre/a_model_student_nn.pkl", mode='wb'))

                #if(modeltype == 'lg'):
                   #pickle.dump(a_base_learners_lg, open("pre/a_model_student_lg.pkl", mode='wb'))

            status.Unindent()

        else:
            # Load the pre-trained models
            status.Print("Loading pre-trained student implementation Scheme A...")
            a_base_learners_svm = pickle.load(open("pre/a_model_student_svm.pkl", mode='rb'))
            a_base_learners_nn = pickle.load(open("pre/a_model_student_nn.pkl", mode='rb'))
            a_base_learners_lg = pickle.load(open("pre/a_model_student_lg.pkl", mode='rb'))

        status.Print("Evaluating the student implementation against Scheme A...")

        # Evaluate the dataset against each of the base learners
        frs_bagging_binary_score = None
        frn_bagging_binary_score = None
        frl_bagging_binary_score = None

        for a_model_i in range(3):
            modeltype = a_modeltypes[a_model_i]
            
            message = "Evaluating dataset against {} student implementation base learners...".format(modeltype)
            status.Print(message)

            base_learner = None

            if(modeltype == 'svm'):
                base_learner = a_base_learners_svm
            elif(modeltype == 'nn'):
                base_learner = a_base_learners_nn
            elif(modeltype == 'lg'):
                base_learner = a_base_learners_lg

            fr_results = [0] * a_base_learner_count
            
            status.Indent()
            for a_base_learner_i in np.arange(a_base_learner_count):

                y_pred = None

                if(modeltype == 'svm'):
                    fr_results[a_base_learner_i] = base_learner[a_base_learner_i].predict(a_x_test_tensor_svm).detach().numpy()
                    y_pred = np.asarray([1 if element > 0 else 0 for element in fr_results[a_base_learner_i]])
                else:
                    fr_results[a_base_learner_i] = base_learner[a_base_learner_i].predict(a_x_test_tensor).detach().numpy()
                    y_pred = np.asarray([1 if element > 0.5 else 0 for element in fr_results[a_base_learner_i]])

                # Print the results of the current base learner for convenience
                message_evaluation = FEBRL.evaluation(a_y_test, y_pred)
                if(args.verbose == True):
                    message = "Base learner {}: {}".format(a_base_learner_i, message_evaluation)
                else:
                    message = "Base learner {}: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(str(a_base_learner_i),
                                message_evaluation["precision"],
                                message_evaluation["sensitivity"],
                                message_evaluation["F-score"])

                status.Print(message, prependTimestamp=False)
  
            fr_bagging_raw_score = np.average(fr_results, axis=0)
            fr_bagging_binary_score = np.copy(fr_bagging_raw_score)

            if(modeltype == 'svm'):
                fr_bagging_binary_score[fr_bagging_binary_score > 0] = 1
                fr_bagging_binary_score[fr_bagging_binary_score <= 0] = 0
            else:
                fr_bagging_binary_score[fr_bagging_binary_score > 0.5] = 1
                fr_bagging_binary_score[fr_bagging_binary_score <= 0.5] = 0

            fr_bagging_evaluation = FEBRL.evaluation(a_y_test, fr_bagging_binary_score)
            
            if(args.verbose == True):
                message = "{} bagging: {}".format(modeltype, fr_bagging_evaluation)
            else:
                message = "{} bagging: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(modeltype,
                            fr_bagging_evaluation["precision"],
                            fr_bagging_evaluation["sensitivity"],
                            fr_bagging_evaluation["F-score"])

            status.Print(message, prependTimestamp=False)
            print("")

            status.Unindent()

            if(modeltype == 'svm'):
                frs_bagging_binary_score = copy.deepcopy(fr_bagging_binary_score)
            elif(modeltype == 'nn'):
                frn_bagging_binary_score = copy.deepcopy(fr_bagging_binary_score)
            elif(modeltype == 'lg'):
                frl_bagging_binary_score = copy.deepcopy(fr_bagging_binary_score)

        fr_stacking_threshold = 0.99

        fr_stacking_binary_score = np.average([frs_bagging_binary_score, frn_bagging_binary_score, frl_bagging_binary_score], axis=0)
        fr_stacking_binary_score[fr_stacking_binary_score > fr_stacking_threshold] = 1
        fr_stacking_binary_score[fr_stacking_binary_score <= fr_stacking_threshold] = 0
        fr_stacking_evaluation = FEBRL.evaluation(a_y_test, fr_stacking_binary_score)
        
        status.Print("Student Implementation Scheme A stacking performance:")

        if(args.verbose == True):
            message = "{}".format(fr_stacking_evaluation)
        else:
            message = "precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(
                        fr_stacking_evaluation["precision"],
                        fr_stacking_evaluation["sensitivity"],
                        fr_stacking_evaluation["F-score"])

        status.Print(message, prependTimestamp=False)
        print("")

        if(args.search == True):
            print("")
            status.Print("===== HYPERPARAMETER SEARCH (STUDENT IMPLEMENTATION)=====")

            # Perform SVM base learner evaluation using the hyperparameter search range provided by the original paper
            frs_inverse_reg_range = [.001,.002,.005,.01,.02,.05,.1,.2,.5,1,5,10,20,50,100,200,500,1000,2000,5000]

            status.Print("Performing hyperparameter search for student implementation Scheme A...")

            status.Print("Performing hyperparameter search for SVM (SGD L2 penalty)...")
            status.Indent()
            for inverse_reg in frs_inverse_reg_range:
                # Create an instance of the SVM
                febrl_reproducer_svm = fs.FEBRLReproducerSVM(num_features=a_x_train_tensor.shape[1], inverse_reg=inverse_reg)

                # Train the model
                febrl_reproducer_svm.fit(a_x_train_tensor_svm, a_y_train_tensor_svm)

                # Test the model
                frs_output = febrl_reproducer_svm.predict(a_x_test_tensor_svm).detach()

                y_pred = np.asarray([1 if element > 0 else 0 for element in frs_output])

                # Print the results
                message_evaluation = FEBRL.evaluation(a_y_test, y_pred)

                if(args.verbose == True):
                    message = "inverse_reg = {}: {}".format(inverse_reg, message_evaluation)
                else:
                    message = "inverse_reg = {}: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(inverse_reg,
                                message_evaluation["precision"],
                                message_evaluation["sensitivity"],
                                message_evaluation["F-score"])

                status.Print(message, prependTimestamp=False)

            status.Unindent()

            # Perform NN base learner evaluation using the hyperparameter search range provided by the original paper
            frn_weight_decay_range = [.001,.002,.005,.01,.02,.05,.1,.2,.5,1,5,10,20,50,100,200,500,1000,2000,5000]  

            status.Print("Performing hyperparameter search for NN (SGD L2 penalty)...")
            status.Indent()
            for weight_decay in frn_weight_decay_range:
                # Create an instance of the feed-forward neural network
                febrl_reproducer_nn = fs.FEBRLReproducerNN(num_features=a_x_train_tensor.shape[1], weight_decay=weight_decay)

                # Train the model
                febrl_reproducer_nn.fit(a_x_train_tensor, a_y_train_tensor)

                # Test the model
                frn_output = febrl_reproducer_nn.predict(a_x_test_tensor).detach()

                y_pred = np.asarray([1 if element > 0.5 else 0 for element in frn_output])

                # Print the results
                message_evaluation = FEBRL.evaluation(a_y_test, y_pred)

                if(args.verbose == True):
                    message = "weight_decay = {}: {}".format(weight_decay, message_evaluation)
                else:
                    message = "weight_decay = {}: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(weight_decay,
                                message_evaluation["precision"],
                                message_evaluation["sensitivity"],
                                message_evaluation["F-score"])

                status.Print(message, prependTimestamp=False)

            # Perform logistic regression base learner evaluation using the hyperparameter search range provided by the original paper
            frl_inverse_reg_range = [.001,.002,.005,.01,.02,.05,.1,.2,.5,1,5,10,20,50,100,200,500,1000,2000,5000]

            status.Unindent()

            status.Print("Performing hyperparameter search for Logistic Regression (SGD L2 penalty)...")
            status.Indent()
            for inverse_reg in frl_inverse_reg_range:
                # Create an instance of the logistic regression model
                febrl_reproducer_lr = fs.FEBRLReproducerLR(num_features=a_x_train_tensor.shape[1], inverse_reg=inverse_reg)

                # Train the model
                febrl_reproducer_lr.fit(a_x_train_tensor, a_y_train_tensor)

                # Test the model
                frl_output = febrl_reproducer_lr.predict(a_x_test_tensor).detach()

                y_pred = np.asarray([1 if element > 0.5 else 0 for element in frl_output])

                if(args.verbose == True):
                    message = "weight_decay = {}: {}".format(inverse_reg, message_evaluation)
                else:
                    message = "weight_decay = {}: precision: {:06.4f}, sensitivity: {:06.4f}, F-score: {:06.4f}".format(inverse_reg,
                                message_evaluation["precision"],
                                message_evaluation["sensitivity"],
                                message_evaluation["F-score"])

                status.Print(message, prependTimestamp=False)

            status.Unindent()
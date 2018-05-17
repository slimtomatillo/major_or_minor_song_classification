# Standard libraries
import pandas as pd

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Performance metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

# Logging
import logging
logging.basicConfig(level=logging.INFO)

from transform import transform


def model(df_filename):
    """
    Function that reads in a dataframe
    and builds and evaluates several
    models using that data

    Models:
    Logistic Regression
    Logistic Regression + Grid Search
    KNN
    KNN + Grid Search
    SVC
    SVC Normalized
    SVC Normalized + Grid Search
    SVC Normalized + Random Search
    Naive Bayes
    Decision Tree
    Decision Tree + Grid Search
    Random Forest
    Random Forest + Random Search

    Performance:
    Accuracy

    :param df_filename: str

    :return: df of models and performance:
    pandas.DataFrame
    """
    data, data_norm = transform(df_filename)

    logging.info('Dataframes received.')

    # Create list of model and accuracy dicts
    list_of_perform = []

    # Define X, y
    X = data.iloc[:,1:9]
    y = data.iloc[:,9]

    # Split data into train, holdout split
    X_tr, X_holdout, y_tr, y_holdout = train_test_split(X, y, test_size=0.3, random_state=1)

    # Split train into train, test split
    X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.3, random_state=1)

    # Define X, y
    X_norm = data_norm.iloc[:,:8]
    y_norm = data_norm.iloc[:,8]

    # Split data into train, holdout split
    X_tr_norm, X_holdout_norm, y_tr_norm, y_holdout_norm = train_test_split(X_norm, y_norm, test_size=0.3,
                                                                            random_state=1)

    # Split data into train, test split
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_tr_norm, y_tr_norm, test_size=0.3,
                                                                            random_state=1)

    logging.info('Train, holdout splits created.')

    # Create list of model and accuracy dicts
    list_of_perform = []

    # List to keep track of models
    models = []

    # List to keep track of y_scores
    y_scores = []

    logging.info('Creating Logistic Regression models.')

    # LOGISTIC REGRESSION

    # Baseline
    # Instantiate model
    lr_baseline = LogisticRegression(random_state=1)

    # Calculate accuracy of train
    lr_base_acc = cross_val_score(lr_baseline, X_train, y_train).mean()
    lr_base_std = cross_val_score(lr_baseline, X_train, y_train).std()
    logging.info('Accuracy of log reg baseline (TRAIN): {} +/- {}'.format(round(lr_base_acc, 2), round(lr_base_std, 2)))

    # Fit model
    lr_baseline.fit(X_train, y_train)

    # Get predictions and probabilities
    lr_base_preds = lr_baseline.predict(X_test)
    lr_base_y_score = lr_baseline.predict_proba(X_test)

    # Calculate accuracy of test
    lr_base_acc_test = round(accuracy_score(y_test, lr_base_preds), 2)
    logging.info(f'Accuracy of log reg baseline (TEST): {lr_base_acc_test}')

    # Get precision, recall, f1-score, and support
    precision, recall, fscore, support = score(y_test, lr_base_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'LogReg Base'),
        ('Train Accuracy', round(lr_base_acc, 2)),
        ('Test Accuracy', lr_base_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('LogReg Base')

    # Add y_score to list
    y_scores.append(lr_base_y_score)

    # Grid Search logistic regression
    # Define the parameter values that should be searched
    C_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 40, 50, 60, 100, 1000, 10000]
    fit_intercept_range = [True, False]

    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid = dict(C=C_range,
                      fit_intercept=fit_intercept_range, )

    logging.info('Grid Searching Logistic Regression hyper-parameters...')

    # Instantiate and fit the grid
    lr_grid = GridSearchCV(lr_baseline, param_grid)
    lr_grid.fit(X_train, y_train)

    # View best parameter
    logging.info('Best parameters found:')
    logging.info(lr_grid.best_params_)

    # Logistic Regression optimized w/ Grid Search
    # Instantiate model
    lr_opt = LogisticRegression(C=100, fit_intercept=True, random_state=1)

    # Calculate accuracy of train
    lr_opt_acc = cross_val_score(lr_opt, X_train, y_train).mean()
    lr_opt_std = cross_val_score(lr_opt, X_train, y_train).std()
    logging.info('Accuracy of log reg optimized (TRAIN): {} +/- {}'.format(round(lr_opt_acc, 2), round(lr_opt_std, 2)))

    # Fit model
    lr_opt.fit(X_train, y_train)

    # Get predictions and probabilities
    lr_opt_preds = lr_opt.predict(X_test)
    lr_opt_y_score = lr_opt.predict_proba(X_test)

    # Calculate accuracy of test
    lr_opt_acc_test = round(accuracy_score(y_test, lr_opt_preds), 2)
    logging.info(f'Accuracy of log reg optimized (TEST): {lr_opt_acc_test}')

    # Get precision, recall, f1-score, and support
    precision, recall, fscore, support = score(y_test, lr_opt_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'LogReg Opt GS'),
        ('Train Accuracy', round(lr_opt_acc, 2)),
        ('Test Accuracy', lr_opt_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('LogReg Opt GS')

    # Add y_score to list
    y_scores.append(lr_opt_y_score)

    logging.info('Creating KNN models.')

    # KNN

    # Baseline
    # Instantiate model
    neigh_base = KNeighborsClassifier()

    # Calculate accuracy of train
    neigh_base_acc = cross_val_score(neigh_base, X_train, y_train).mean()
    neigh_base_std = cross_val_score(neigh_base, X_train, y_train).std()
    logging.info('Accuracy of KNN baseline (TRAIN): {} +/- {}'.format(round(neigh_base_acc, 2), round(neigh_base_std, 2)))

    # Fit model
    neigh_base.fit(X_train, y_train)

    # Get predictions and probabilities
    neigh_base_preds = neigh_base.predict(X_test)
    neigh_base_y_score = neigh_base.predict_proba(X_test)

    # Calculate accuracy of test
    neigh_base_acc_test = round(accuracy_score(y_test, neigh_base_preds), 2)
    logging.info(f'Accuracy of KNN baseline (TEST): {neigh_base_acc_test}')

    # Get precision, recall, f1-score, and support
    precision, recall, fscore, support = score(y_test, neigh_base_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'KNN Base'),
        ('Train Accuracy', round(neigh_base_acc, 2)),
        ('Test Accuracy', neigh_base_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('KNN Base')

    # Add y_score to list
    y_scores.append(neigh_base_y_score)

    # Grid Search KNN
    # Define parameters that should be searched
    n_neighbors_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights_range = ['uniform', 'distance']
    algorithm_range = ['ball_tree', 'kd_tree', 'brute']
    p_range = [1, 2, 3]

    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid_2 = dict(n_neighbors=n_neighbors_range,
                        weights=weights_range,
                        algorithm=algorithm_range,
                        p=p_range
                        )

    logging.info('Grid Searching KNN hyper-parameters...')

    # Instantiate and fit the grid
    knn_grid = GridSearchCV(neigh_base, param_grid_2)
    knn_grid.fit(X_train, y_train)

    logging.info('Best parameters found:')
    logging.info(knn_grid.best_params_)

    # KNN optimized w/ Grid Search
    # Instantiate the model
    knn_opt = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', p=1)

    # Calculate accuracy of train
    knn_opt_acc = cross_val_score(knn_opt, X_train, y_train).mean()
    knn_opt_std = cross_val_score(knn_opt, X_train, y_train).std()
    logging.info('Accuracy of KNN optimized (TRAIN): {} +/- {}'.format(round(knn_opt_acc, 2), round(knn_opt_std, 2)))

    # Fit model
    knn_opt.fit(X_train, y_train)

    # Get predictions and probabilities
    knn_opt_preds = knn_opt.predict(X_test)
    knn_opt_y_score = knn_opt.predict_proba(X_test)

    # Calculate accuracy of test
    knn_opt_acc_test = round(accuracy_score(y_test, knn_opt_preds), 2)
    logging.info(f'Accuracy of KNN optimized (TEST): {knn_opt_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, knn_opt_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'KNN Opt GS'),
        ('Train Accuracy', round(knn_opt_acc, 2)),
        ('Test Accuracy', knn_opt_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('KNN Opt GS')

    # Add y_score to list
    y_scores.append(knn_opt_y_score)

    logging.info('Creating SVC models.')

    # SVC

    # Baseline
    # Instantiate the model
    svc_base = SVC(probability=True, random_state=1)

    # Calculate accuracy of train
    svc_base_acc = cross_val_score(svc_base, X_train, y_train).mean()
    svc_base_std = cross_val_score(svc_base, X_train, y_train).std()
    logging.info('Accuracy of SVC baseline (TRAIN): {} +/- {}'.format(round(svc_base_acc, 2), round(svc_base_std, 2)))

    # Fit model
    svc_base.fit(X_train, y_train)

    # Get predictions and probabilities
    svc_base_preds = svc_base.predict(X_test)
    svc_base_y_score = svc_base.predict_proba(X_test)

    # Calculate accuracy of test
    svc_base_acc_test = round(accuracy_score(y_test, svc_base_preds), 2)
    logging.info(f'Accuracy of SVC baseline (TEST): {svc_base_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, svc_base_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'SVC Base'),
        ('Train Accuracy', round(svc_base_acc, 2)),
        ('Test Accuracy', svc_base_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('SVC Base')

    # Add y_score to list
    y_scores.append(svc_base_y_score)

    # Normalized Baseline
    # Instantiate the model
    svc_base_norm = SVC(probability=True, random_state=1)

    # Calculate accuracy of train
    svc_base_norm_acc = cross_val_score(svc_base_norm, X_train_norm, y_train_norm).mean()
    svc_base_norm_std = cross_val_score(svc_base_norm, X_train_norm, y_train_norm).std()
    logging.info('Accuracy of SVC normalized baseline (TRAIN): {} +/- {}'.format(round(svc_base_norm_acc, 2),
                                                                  round(svc_base_norm_std, 2)))

    # Fit model
    svc_base_norm.fit(X_train_norm, y_train_norm)

    # Get predictions and probabilities
    svc_base_norm_preds = svc_base_norm.predict(X_test_norm)
    svc_base_norm_y_score = svc_base_norm.predict_proba(X_test_norm)

    # Calculate accuracy of test
    svc_base_norm_acc_test = round(accuracy_score(y_test_norm, svc_base_norm_preds), 2)
    logging.info(f'Accuracy of SVC normalized baseline (TEST): {svc_base_norm_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test_norm, svc_base_norm_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'SVC Norm Base'),
        ('Train Accuracy', round(svc_base_norm_acc, 2)),
        ('Test Accuracy', svc_base_norm_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('SVC Norm Base')

    # Add y_score to list
    y_scores.append(svc_base_norm_y_score)

    # Grid Search SVC
    # Define parameters that should be searched
    C_range = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 50, 100, 1000]
    kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']
    tol_range = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid_3 = dict(C=C_range,
                        kernel=kernel_range,
                        tol=tol_range
                        )

    # Instantiate the grid
    svc_grid = GridSearchCV(svc_base_norm, param_grid_3)

    logging.info('Grid Searching SVC hyper-parameters...')

    # Fit the model
    svc_grid.fit(X_train_norm, y_train_norm)

    logging.info('Best parameters found:')
    logging.info(svc_grid.best_params_)

    # SVC optimized w/ Grid Search
    # Instantiate the model
    svc_opt = SVC(C=50, kernel='poly', tol=0.3, probability=True, random_state=1)

    # Calculate accuracy of train
    svc_opt_acc = cross_val_score(svc_opt, X_train_norm, y_train_norm).mean()
    svc_opt_std = cross_val_score(svc_opt, X_train_norm, y_train_norm).std()
    logging.info('Accuracy of SVC optimized w/ GridSearch (TRAIN): {} +/- {}'.format(round(svc_opt_acc, 2), round(svc_opt_std, 2)))

    # Fit model
    svc_opt.fit(X_train_norm, y_train_norm)

    # Get predictions and probabilities
    svc_opt_preds = svc_opt.predict(X_test_norm)
    svc_opt_y_score = svc_opt.predict_proba(X_test_norm)

    # Calculate accuracy of test
    svc_opt_acc_test = round(accuracy_score(y_test_norm, svc_opt_preds), 2)
    logging.info(f'Accuracy of SVC optimized w/ GridSearch (TEST): {svc_opt_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test_norm, svc_opt_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'SVC Opt GS'),
        ('Train Accuracy', round(svc_opt_acc, 2)),
        ('Test Accuracy', svc_opt_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('SVC Opt GS')

    # Add y_score to list
    y_scores.append(svc_opt_y_score)

    # Random Search SVC
    # Define parameters that should be searched
    param_rand = {"C": [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 50, 100, 1000, 10000],
                  "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                  "tol": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}

    # Choose number of searches
    n_iter_search = 40

    # Instantiate the model
    svc_random_search = RandomizedSearchCV(svc_base_norm, param_distributions=param_rand, n_iter=n_iter_search)

    logging.info('Random Searching SVC hyper-parameters...')

    # Fit the model
    svc_random_search.fit(X_train_norm, y_train_norm)

    logging.info('Best parameters found:')
    logging.info(svc_random_search.best_params_)

    # SVC optimized w/ Random Search
    # Instantiate the model
    svc_rand = SVC(C=1000, kernel='rbf', tol=0.001, probability=True, random_state=1)

    # Calculate accuracy of train
    svc_rand_acc = cross_val_score(svc_rand, X_train_norm, y_train_norm).mean()
    svc_rand_std = cross_val_score(svc_rand, X_train_norm, y_train_norm).std()
    logging.info('Accuracy of SVC optimized w/ RandomSearch (TRAIN): {} +/- {}'.format(round(svc_rand_acc, 2), round(svc_rand_std, 2)))

    # Fit model
    svc_rand.fit(X_train_norm, y_train_norm)

    # Get predictions and probabilities
    svc_rand_preds = svc_rand.predict(X_test_norm)
    svc_rand_y_score = svc_rand.predict_proba(X_test_norm)

    # Calculate accuracy of test
    svc_rand_acc_test = round(accuracy_score(y_test_norm, svc_rand_preds), 2)
    logging.info(f'Accuracy of SVC optimized w/ RandomSearch (TEST): {svc_rand_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test_norm, svc_rand_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'SVC Opt RS'),
        ('Train Accuracy', round(svc_rand_acc, 2)),
        ('Test Accuracy', svc_rand_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('SVC Opt RS')

    # Add y_score to list
    y_scores.append(svc_rand_y_score)

    logging.info('Creating Naive Bayes model.')

    # NAIVE BAYES CLASSIFIER

    # Instantiate the model
    nbc_base = GaussianNB()

    # Calculate accuracy of train
    nbc_base_acc = cross_val_score(nbc_base, X_train, y_train).mean()
    nbc_base_std = cross_val_score(nbc_base, X_train, y_train).std()
    logging.info('Accuracy of Naive Bayes Classifier (TRAIN): {} +/- {}'.format(round(nbc_base_acc, 2), round(nbc_base_std, 2)))

    # Fit model
    nbc_base.fit(X_train, y_train)

    # Get predictions and probabilities
    nbc_base_preds = nbc_base.predict(X_test)
    nbc_base_y_score = nbc_base.predict_proba(X_test)

    # Calculate accuracy of test
    nbc_base_acc_test = round(accuracy_score(y_test, nbc_base_preds), 2)
    logging.info(f'Accuracy of Naive Bayes Classifier (TEST): {nbc_base_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, nbc_base_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'Naive Bayes'),
        ('Train Accuracy', round(nbc_base_acc, 2)),
        ('Test Accuracy', nbc_base_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('Naive Bayes')

    # Add y_score to list
    y_scores.append(nbc_base_y_score)

    logging.info('Creating Decision Tree models.')

    # DECISION TREE CLASSIFIER

    # Baseline
    # Instantiate the model
    dt_base = DecisionTreeClassifier(random_state=1)

    # Calculate accuracy of train
    dt_base_acc = cross_val_score(dt_base, X_train, y_train).mean()
    dt_base_std = cross_val_score(dt_base, X_train, y_train).std()
    logging.info('Accuracy of Decision Tree baseline (TRAIN): {} +/- {}'.format(round(dt_base_acc, 2), round(dt_base_std, 2)))

    # Fit model
    dt_base.fit(X_train, y_train)

    # Get predictions and probabilities
    dt_base_preds = dt_base.predict(X_test)
    dt_base_y_score = dt_base.predict_proba(X_test)

    # Calculate accuracy of test
    dt_base_acc = round(accuracy_score(y_test, dt_base_preds), 2)
    logging.info(f'Accuracy of Decision Tree baseline (TEST): {dt_base_acc}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, dt_base_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'DecTree Base'),
        ('Train Accuracy', round(dt_base_acc, 2)),
        ('Test Accuracy', dt_base_acc),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('DecTree Base')

    # Add y_score to list
    y_scores.append(dt_base_y_score)

    # Grid Search Decision Tree
    # Define parameters that should be searched
    criterion_range = ['gini', 'entropy']
    splitter_range = ['best', 'random']
    max_depth_range = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_split_range = [10, 20, 50, 100, 500]
    min_samples_leaf_range = [10, 20, 50, 100, 250, 500]

    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid_4 = dict(criterion=criterion_range,
                        splitter=splitter_range,
                        max_depth=max_depth_range,
                        min_samples_split=min_samples_split_range,
                        min_samples_leaf=min_samples_leaf_range
                        )

    logging.info('Grid Searching Decision Tree hyper-parameters...')

    # Instantiate the grid
    dt_grid = GridSearchCV(dt_base, param_grid_4)

    # Fit the model
    dt_grid.fit(X_train, y_train)

    logging.info('Best parameters found:')
    logging.info(dt_grid.best_params_)

    # Decision Tree optimized w/ Grid Search

    # Instantiate the model
    dt_opt = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=10,
                                    min_samples_leaf=10, random_state=1)

    # Calculate accuracy of train
    dt_opt_acc = cross_val_score(dt_opt, X_train, y_train).mean()
    dt_opt_std = cross_val_score(dt_opt, X_train, y_train).std()
    logging.info('Accuracy of Decision Tree optimized (TRAIN): {} +/- {}'.format(round(dt_opt_acc, 2), round(dt_opt_std, 2)))

    # Fit model
    dt_opt.fit(X_train, y_train)

    # Get predictions and probabilities
    dt_opt_preds = dt_opt.predict(X_test)
    dt_opt_y_score = dt_opt.predict_proba(X_test)

    # Calculate accuracy of test
    dt_opt_acc_test = round(accuracy_score(y_test, dt_opt_preds), 2)
    logging.info(f'Accuracy of Decision Tree optimized (TEST): {dt_opt_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, dt_opt_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'DecTree Opt GS'),
        ('Train Accuracy', round(dt_opt_acc, 2)),
        ('Test Accuracy', dt_opt_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('DecTree Opt GS')

    # Add y_score to list
    y_scores.append(dt_opt_y_score)

    # View feature importances
    dt_feat_import = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt_opt.feature_importances_
    })

    logging.info(dt_feat_import)

    logging.info('Creating Random Forest models.')

    # RANDOM FOREST CLASSIFIER

    # Baseline
    # Instantiate the model
    rf_base = RandomForestClassifier(random_state=1)

    # Calculate accuracy of train
    rf_base_acc = cross_val_score(rf_base, X_train, y_train).mean()
    rf_base_std = cross_val_score(rf_base, X_train, y_train).std()
    logging.info('Accuracy of Random Forest baseline (TRAIN): {} +/- {}'.format(round(rf_base_acc, 2), round(rf_base_std, 2)))

    # Fit model
    rf_base.fit(X_train, y_train)

    # Get predictions and probabilities
    rf_base_preds = rf_base.predict(X_test)
    rf_base_y_score = rf_base.predict_proba(X_test)

    # Calculate accuracy of test
    rf_base_acc_test = round(accuracy_score(y_test, rf_base_preds), 2)
    logging.info(f'Accuracy of Random Forest baseline (TEST): {rf_base_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, rf_base_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'RandFor Base'),
        ('Train Accuracy', round(rf_base_acc, 2)),
        ('Test Accuracy', rf_base_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('RandFor Base')

    # Add y_score to list
    y_scores.append(rf_base_y_score)

    # Random Search Random Forest
    # Define parameters that should be searched
    n_estimators_range = [x for x in range(1, 31)]
    criterion_range = ['gini', 'entropy']
    max_features_range = [None, 'auto', 'log2', 1, 2, 3, 4, 5, 6, 7, 8]
    max_depth_range = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_split_range = [10, 20, 50, 100, 500]
    min_samples_leaf_range = [10, 20, 50, 100, 250, 500]
    bootstrap_range = [True, False]

    # Create a parameter grid: map the parameter names to the values that should be searched
    param_rand_rf = dict(n_estimators=n_estimators_range,
                         criterion=criterion_range,
                         max_features=max_features_range,
                         max_depth=max_depth_range,
                         min_samples_split=min_samples_split_range,
                         min_samples_leaf=min_samples_leaf_range,
                         bootstrap=bootstrap_range
                         )

    # Choose number of searches
    n_iter_search = 40

    logging.info('Random Searching Random Forest hyper-parameters...')

    # Instantiate the model
    rf_random_search = RandomizedSearchCV(rf_base, param_distributions=param_rand_rf, n_iter=n_iter_search)

    # Fit the model
    rf_random_search.fit(X_train, y_train)

    logging.info('Best parameters found:')
    logging.info(rf_random_search.best_params_)

    # Random Forest optimized w/ Random Search
    # Instantiate the model
    rf_opt = RandomForestClassifier(n_estimators=28,
                                    criterion='gini',
                                    max_features=4,
                                    max_depth=4,
                                    min_samples_split=100,
                                    min_samples_leaf=10,
                                    bootstrap=False,
                                    random_state=1)

    # Calculate the accuracy of train
    rf_opt_acc = cross_val_score(rf_opt, X_train, y_train).mean()
    rf_opt_std = cross_val_score(rf_opt, X_train, y_train).std()
    logging.info('Accuracy of Random Forest optimized (TRAIN): {} +/- {}'.format(round(rf_opt_acc, 2), round(rf_opt_std, 2)))

    # Fit model
    rf_opt.fit(X_train, y_train)

    # Get predictions and probabilities
    rf_opt_preds = rf_base.predict(X_test)
    rf_opt_y_score = rf_base.predict_proba(X_test)

    # Calculate accuracy of test
    rf_opt_acc_test = round(accuracy_score(y_test, rf_opt_preds), 2)
    logging.info(f'Accuracy of Random Forest optimized (TEST): {rf_opt_acc_test}')

    # Get precision, recall, f1-score
    precision, recall, fscore, support = score(y_test, rf_opt_preds, average='macro')

    logging.info(f'Precision : {precision}')
    logging.info(f'Recall    : {recall}')
    logging.info(f'F-score   : {fscore}')

    # Add model and accuracy dict to list
    list_of_perform.append(dict([
        ('Model', 'RandFor Opt RS'),
        ('Train Accuracy', round(rf_opt_acc, 2)),
        ('Test Accuracy', rf_opt_acc_test),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(fscore, 2))
    ]))

    # Add model to list
    models.append('RandFor Opt RS')

    # Add y_score to list
    y_scores.append(rf_opt_y_score)

    # View feature importances
    rf_feat_import = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_opt.feature_importances_
    })

    logging.info(rf_feat_import)
    logging.info('Finished!')

    # Turn dict of models and accuracies into pandas df
    model_perf = pd.DataFrame(data=list_of_perform)
    model_perf = model_perf[['Model', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1']]

    logging.info('Models and performance:')
    logging.info(model_perf)

    # Test best 3 models on holdout data
    logging.info('Best models to test with Holdout set:')
    logging.info('1. Decision Tree - Optimized w/ Grid Search')
    logging.info('2. Random Forest - Optimized w/ Random Search')
    logging.info('3. Support Vector Classifier - Optimized w/ Random Search')

    # Keep track of holdout accuracies
    holdout_acc = []

    # 1. Decision Tree - Optimized w/ Grid Search
    # Get predictions and probabilities
    dt_holdout_preds = dt_opt.predict(X_holdout)
    dt_holdout_y_score = dt_opt.predict_proba(X_holdout)

    # Use X_holdout, y_holdout to calculate accuracy
    dt_holdout_acc = round(accuracy_score(y_holdout, dt_holdout_preds), 2)
    logging.info(f'Accuracy of Decision Tree Opt (holdout): {dt_holdout_acc}')

    # Add model and accuracy dict to list
    holdout_acc.append(dict([
        ('Model', 'Decision Tree GS'),
        ('Holdout Accuracy', dt_holdout_acc)
    ]))

    # 2. Random Forest - Optimized w/ Random Search
    # Get predictions and probabilities
    rf_holdout_preds = rf_opt.predict(X_holdout)
    rf_holdout_y_score = rf_opt.predict_proba(X_holdout)

    # Use X_holdout, y_holdout to calculate accuracy
    rf_holdout_acc = round(accuracy_score(y_holdout, rf_holdout_preds), 2)
    logging.info(f'Accuracy of Random Forest Opt (holdout): {rf_holdout_acc}')

    # Add model and accuracy dict to list
    holdout_acc.append(dict([
        ('Model', 'Random Forest RS'),
        ('Holdout Accuracy', rf_holdout_acc)
    ]))

    # 3. Support Vector Classifier - Optimized w/ Random Search
    # Get predictions and probabilities
    svc_holdout_preds = svc_opt.predict(X_holdout_norm)
    svc_holdout_y_score = svc_opt.predict_proba(X_holdout_norm)

    # Use X_holdout_norm, y_holdout_norm to calculate accuracy
    svc_holdout_acc = round(accuracy_score(y_holdout_norm, svc_holdout_preds), 2)
    print(f'Accuracy of SVC Opt (holdout): {svc_holdout_acc}')

    # Add model and accuracy dict to list
    holdout_acc.append(dict([
        ('Model', 'SVC RS'),
        ('Holdout Accuracy', svc_holdout_acc)
    ]))

    # Reorder columns and rows
    holdout_results = pd.DataFrame(data=holdout_acc, columns=['Model', 'Holdout Accuracy'])
    holdout_results = holdout_results.reindex(index=[2, 0, 1])
    holdout_results = holdout_results.reset_index(drop=True)

    print(holdout_results)

    return(model_perf, holdout_results)

#model('/Users/alexandrasmith/ds/metis/proj3_mcnulty/TEST/30_sec_data.csv')
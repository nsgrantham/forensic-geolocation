import os
import click

import numpy as np
import pandas as pd
from scipy import stats

from biom import load_table

from models.points import SpatialPoints
from models.bda import GaussianKernelSmoother, BayesianDiscriminantAnalysis
from models.deepspace import SpatialClassifier, Geolocator

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(881)  # for reproducibility

@click.command()
@click.option('--folds', type=int, 
        help="Number of folds for cross-validation.")
@click.option('--partitions', type=int, 
        help="Number of partition-classifier pairs for DeepSpace algorithm.")
@click.option('--seeds', type=click.Choice(['coarse', 'fine', 'mixed', 'none']),
        help="Density of seeds per partition.")
@click.option('--region', type=float, multiple=True, 
        help="Probability of prediction region to calculate.")
@click.option('--taxa-threshold', type=float, default=0.05, 
        help='Remove taxa that are not present in at least this proportion of samples.')
@click.option('--domain-fp')
@click.option('--area', multiple=True)
@click.option('--centroids')
@click.option('--centroids-fp')
@click.option('--weight-by')
@click.option('--weight-by-fp')   
@click.option('--weight-by-col')
@click.option('--knn-n-neighbors', default=10)
@click.option('--rf-n-estimators', default=100)
@click.option('--nn-epochs', default=20)
@click.option('--nn-batch-size', default=64)
@click.option('--nn-verbose', is_flag=True)
@click.option('--dnn-epochs', default=20)
@click.option('--dnn-batch-size', default=64)
@click.option('--dnn-verbose', is_flag=True)
@click.option('--area-clf-epochs', default=20)
@click.option('--area-clf-batch-size', default=64)
@click.option('--area-clf-verbose', is_flag=True)
@click.argument('input_')
@click.argument('output')
def cross_validate(input_, output, folds, partitions, seeds, region, domain_fp,
        area, centroids, centroids_fp, weight_by, weight_by_fp, weight_by_col,
        knn_n_neighbors, rf_n_estimators, nn_epochs, nn_batch_size, nn_verbose,
        dnn_epochs, dnn_batch_size, dnn_verbose, area_clf_epochs, 
        area_clf_batch_size, area_clf_verbose, taxa_threshold):
    
    # First, define the models to be cross-validated

    def fit_knn(sp, partition, **kwargs):
        clf = KNeighborsClassifier(n_neighbors=knn_n_neighbors, metric='jaccard')
        sp_clf = SpatialClassifier(clf, partition)
        sp_clf.fit(sp)
        return sp_clf
        
    def fit_rf(sp, partition, sample_weight):
        clf = RandomForestClassifier(n_estimators=rf_n_estimators)
        sp_clf = SpatialClassifier(clf, partition)
        sp_clf.fit(sp, sample_weight=sample_weight)
        return sp_clf
        
    def fit_nn(sp, partition, sample_weight):
        model = Sequential()
        model.add(Dense(2048, input_shape=(sp.shape[1], )))
        model.add(Activation('relu'))
        model.add(Dense(partition.shape[0]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), 
                metrics=['accuracy'])
        sp_clf = SpatialClassifier(model, partition)
        sp_clf.fit(sp, to_categorical=True, sample_weight=sample_weight, 
                epochs=nn_epochs, batch_size=nn_batch_size, verbose=nn_verbose)
        return sp_clf
    
    def fit_dnn(sp, partition, sample_weight):
        model = Sequential()
        model.add(Dense(2048, input_shape=(sp.shape[1], )))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(partition.shape[0]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), 
                metrics=['accuracy'])
        clf = SpatialClassifier(model, partition)
        clf.fit(sp, to_categorical=True, sample_weight=sample_weight, 
                epochs=dnn_epochs, batch_size=dnn_batch_size, verbose=dnn_verbose)
        return clf
    
    def fit_area_clf(X, Y, sample_weight):
        model = Sequential()
        model.add(Dense(2048, input_shape=(X.shape[1], )))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(Y.shape[1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        model.fit(X, Y, sample_weight=sample_weight, epochs=area_clf_epochs, 
                batch_size=area_clf_batch_size, verbose=area_clf_verbose)
        return model
    
    model_fitters = {
            'Spatial KNN': fit_knn,
            'Spatial RF': fit_rf,
            'Spatial NN': fit_nn,
            'DeepSpace': fit_dnn,
            }
    
    click.echo("Loading domain file")
    df = pd.read_csv(domain_fp, index_col='domain')
    coord_names = ['lat', 'lon']
    area_names = list(area)
    domain = SpatialPoints(coords=df[coord_names], areas=df[area_names])
    
    if seeds == 'none':
        click.echo("No seeds, performing {} classification".format(centroids))
        click.echo("Loading centroids from file")
        area_centroids = pd.read_csv(centroids_fp, index_col=centroids)

    # Load biom data
    table = load_table(input_).pa()  # presence/absence
    eliminate_exceptionally_rare_taxa = lambda values, id_, md: np.mean(values > 0) > taxa_threshold
    table.filter(eliminate_exceptionally_rare_taxa, axis='observation')
    click.echo("Modeling {} sample points with dimensionality {}.".format(table.shape[1], table.shape[0]))

    # Important variables and metrics to calculate
    regions = list(region)
    regions.sort()
    metrics = ['best_pred', 'error'] + list(map(str, regions))
    add_area_prefix = lambda s: [s + a for a in area_names]
    pred_prefix = 'pred_'
    near_prefix = 'near_'
    true_prefix = 'true_'
    over_prefix = 'over_'
    areas = add_area_prefix(pred_prefix) + add_area_prefix(near_prefix) + \
        add_area_prefix(true_prefix) + add_area_prefix(over_prefix)

    col_names = ['seeds', 'model', 'fold', 'true_lat', 'true_lon', 'pred_lat', 
            'pred_lon'] + metrics + areas
    results = pd.DataFrame(columns=col_names)
    results.index.name = 'ids'
    
    # Set number of partititons and number
    n_samples = table.shape[1]
    seed_vec = [np.round(x * n_samples).astype(int) for x in np.arange(0.05, 0.5, step=0.05)]
    if seeds == 'coarse':
        seed_vec = seed_vec[:2]
    elif seeds == 'fine':
        seed_vec = seed_vec[-2:]
    
    sample_ids = table.ids(axis='sample')
    assigned_fold = dict(zip(sample_ids, np.random.choice(folds, size=len(sample_ids))))
    
    for fold in range(folds):
        click.echo("\n===============\nFold {} of {}\n===============\n".format(fold+1, folds))
        # Allocate points in fold to testing set, all other points to training set
        test_train = lambda id_, md: 'test' if assigned_fold[id_] == fold else 'train'
        sub_tables = dict(table.partition(test_train, axis='sample'))
        sp_train = SpatialPoints.from_biom_table(sub_tables['train'], verbose=True, 
            coord_names=coord_names, area_names=area_names, ids_name='ids')
        sp_test = SpatialPoints.from_biom_table(sub_tables['test'], verbose=True,
            coord_names=coord_names, area_names=area_names, ids_name='ids')
        
        # Determine nearest domain point to every true spatial point.
        # This will be used later to determine if the model predicts the best 
        # possible point in the domain nearest to the point's true origin.
        dist = sp_test.pairwise_distances_from(domain)
        min_dist_pairs = dist.loc[dist.groupby(level=0).idxmin()]
        min_dist_pairs.index.rename(['ids', 'domain'], inplace=True)
        near_domain = pd.Series(list(min_dist_pairs.index.get_level_values('domain')),
                index=min_dist_pairs.index.get_level_values('ids'), name='domain')
        near_areas = (near_domain.to_frame()
                .join(domain.areas, on='domain')
                .drop('domain', axis=1))    
        true_areas = sp_test.areas.copy()
        
        # Weight training set points relative to state population
        if weight_by:
            population = pd.read_csv(weight_by_fp, index_col=weight_by) 
            sample_weights = weight_points_by_population(sp_train, 
                    population[weight_by_col], weight_by).as_matrix()
        else:
            sample_weights = None
        
        # Initialize objects to store model results for this fold
        fold_results = pd.DataFrame(columns=col_names)
        fold_s = pd.Series(str(fold), index=sp_test.ids, name='fold')
        seeds_s = pd.Series(seeds, index=sp_test.ids, name='seeds')
        
        if seeds == 'none':
            
            ## Bayesian discriminant analysis
            subtest_subtrain = lambda id_, md: np.random.choice(['subtrain', 'subtest'], p=[0.9, 0.1])
            sub_sub_tables = dict(sub_tables['train'].partition(subtest_subtrain, axis='sample'))
            sp_subtrain = SpatialPoints.from_biom_table(sub_sub_tables['subtrain'], verbose=True, 
                coord_names=coord_names, area_names=area_names, ids_name='ids')
            sp_subtest = SpatialPoints.from_biom_table(sub_sub_tables['subtest'], verbose=True, 
                coord_names=coord_names, area_names=area_names, ids_name='ids')
            
            # National level bandwidths
            bandwidths = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
            # Local level bandwidths
            #bandwidths = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
            smoothers = [GaussianKernelSmoother(bandwidth) for bandwidth in bandwidths]
            bda = BayesianDiscriminantAnalysis(domain, smoothers)
            
            probs = bda.estimate_occurrence_probabilities(sp_subtrain, domain)
            thresholds = bda.select_threshold(sp_subtest, probs, p=[0.5, 0.75, 0.9])
            likelihood_test = bda.evaluate_likelihood(sp_test.values, probs)
            sp_preds = bda.predict(likelihood_test)
            errors = bda.score(sp_preds, sp_test)
            print(errors.describe())
            
            sp_pred_regs = bda.predict_regions(likelihood_test, thresholds)
            coverage = bda.score_regions(sp_pred_regs, sp_test)
            print(coverage.mean(axis=0))
            
            best_pred = pd.Series(index=sp_test.ids, dtype=bool, name='best_pred')
            pred_areas = pd.DataFrame(index=sp_test.ids, columns=area_names)
            over_areas = pd.DataFrame(index=sp_test.ids, columns=area_names, dtype=bool)
            for id_, sp_pred in sp_preds.items():
                pred_ids = sp_pred['most_likely'].ids
                best_pred.set_value(id_, near_domain.loc[id_] in pred_ids.values)
                pred_ids_areas = (pd.Series(data=pred_ids.values).rename('domain')
                        .to_frame()
                        .join(domain.areas, on='domain')
                        .drop('domain', axis=1))
                for area in area_names:
                    pred_ids_area = pred_ids_areas[area]
                    if pred_ids_area.isnull().all():
                        top_area = np.nan
                    else:
                        top_area = pred_ids_area.value_counts().index[0]
                    pred_areas.set_value(id_, area, top_area)
                    nearest = near_areas[area].loc[id_]
                    is_pred_over_near_area = pred_ids_area.isin([nearest]).any()
                    over_areas.set_value(id_, area, is_pred_over_near_area)
            print(best_pred.to_frame().mean(axis=0))
            print(over_areas.mean(axis=0))
            
            sp_pred_dict = {}
            for id_, sp_pred in sp_preds.items():
                coords = sp_pred['most_likely'].coords
                sp_pred_dict[id_] = {
                        'lat': coords['lat'].iloc[0],
                        'lon': coords['lon'].iloc[0]
                        }
            sp_pred = pd.DataFrame.from_dict(sp_pred_dict, orient='index')
            
            model_s = pd.Series('BDA', index=sp_test.ids, name='model')
            model_results = pd.concat([seeds_s, model_s, fold_s, errors, 
                coverage, best_pred,
                sp_pred.add_prefix(pred_prefix),
                sp_test.coords.add_prefix(true_prefix),
                pred_areas.add_prefix(pred_prefix),
                near_areas.add_prefix(near_prefix),
                true_areas.add_prefix(true_prefix),
                over_areas.add_prefix(over_prefix)], axis=1)
            fold_results = pd.concat([fold_results, model_results], axis=0)
            
            ## Area-level deep neural network
            y = pd.concat([sp_train.areas, sp_test.areas], axis=0)[centroids]
            y_labels, y_levels = y.factorize()
            n_classes = np.max(y_labels) + 1
            y_centroids = area_centroids.loc[y_levels]
            assigned_to = np.array(['test' if assigned_fold[id_] == fold else 'train' for id_ in y.index])
            Y_train = np_utils.to_categorical(y_labels[assigned_to == 'train'], num_classes=n_classes)
            Y_test = np_utils.to_categorical(y_labels[assigned_to == 'test'], num_classes=n_classes)
            X_train = sp_train.values.as_matrix()
            X_test = sp_test.values.as_matrix()
            
            clf = fit_area_clf(X_train, Y_train, sample_weight=sample_weights)
            y_pred = clf.predict_classes(X_test)
            preds = y_centroids.iloc[y_pred]
            preds.index.name = centroids
            preds = preds.reset_index().set_index(sp_test.ids)
            sp_pred = SpatialPoints(coords=preds[['lat', 'lon']])
            errors = sp_pred.distances_from(sp_test).rename('error')
            print(errors.describe())
            
            pred_areas = preds[centroids]
            over_areas = pred_areas == true_areas[centroids]
            print(over_areas.to_frame().mean(axis=0))
            
            model_s = pd.Series('DNN', index=sp_test.ids, name='model')
            model_results = pd.concat([seeds_s, model_s, fold_s, errors, 
                sp_pred.coords.add_prefix(pred_prefix),
                sp_test.coords.add_prefix(true_prefix),
                pred_areas.to_frame().add_prefix(pred_prefix),
                near_areas.add_prefix(near_prefix),
                true_areas.add_prefix(true_prefix),
                over_areas.to_frame().add_prefix(over_prefix)], axis=1)
            fold_results = pd.concat([fold_results, model_results], axis=0)
        
        else:
         
            click.echo("Creating {} Voronoi partitions with {} seeds".format(partitions, seeds))
            n_seeds = np.random.choice(seed_vec, size=partitions)
            parts = [domain.sample(n_seed, reset_index=True) for n_seed in n_seeds]
         
            for model_name, fit_model in model_fitters.items():
                
                # Train a model_name classifier on every partition
                likelihoods = []
                with click.progressbar(parts, label='Fitting {}'.format(model_name)) as bar:
                    for part in bar:
                        sp_clf = fit_model(sp_train, part, sample_weight=sample_weights)
                        likelihood = sp_clf.evaluate_likelihood(sp_test.values, domain)
                        likelihoods.append(likelihood)
               
                # Predict most likely origin and calculate distance from true origin
                geo = Geolocator(domain)
                likelihood_test = pd.concat(likelihoods, axis=1)
                
                #pred_seq = pd.DataFrame(columns = ['ids', 'lat', 'lon', 'n_clf'])
                #for l in range(likelihood_test.shape[1]):
                #    sp_preds_tmp = geo.predict(likelihood_test.iloc[:, 0:(l+1)])
                #    for id_, sp_pred in sp_preds_tmp.items():
                #        coords = sp_pred['most_likely'].coords
                #        tmp = pd.concat([coords,
                #            pd.Series(id_, index=coords.index, name='ids'), 
                #            pd.Series(l+1, index=coords.index, name='n_clf')],
                #            axis=1)
                #        pred_seq = pd.concat([pred_seq, tmp], axis=0)
                #pred_seq.to_csv(os.path.join('models', 'sp_pred.csv'))
                #sp_test.coords.to_csv(os.path.join('models', 'sp_test.csv'))
                
                sp_preds = geo.predict(likelihood_test)
                errors = geo.score(sp_preds, sp_test)
                print(model_name)
                print(errors.describe())
                
                # Calculate probability regions and determine proportion of coverage
                sp_pred_regs = geo.predict_regions(likelihood_test, p=regions)
                coverage = geo.score_regions(sp_pred_regs, sp_test)
                print(coverage.mean(axis=0))
                
                # Finally, check if the nearest domain point to the true origin is 
                # identified as a "most likely" origin (the best possible outcome),
                # determine the areas for the predicted origins (in domain), and
                # check if these areas match the areas of the nearest domain point 
                # to the true origin (indicating a good regional classification).
                best_pred = pd.Series(index=sp_test.ids, dtype=bool, name='best_pred')
                pred_areas = pd.DataFrame(index=sp_test.ids, columns=area_names)
                over_areas = pd.DataFrame(index=sp_test.ids, columns=area_names, dtype=bool)
                for id_, sp_pred in sp_preds.items():
                    pred_ids = sp_pred['most_likely'].ids
                    best_pred.set_value(id_, near_domain.loc[id_] in pred_ids.values)
                    pred_ids_areas = (pd.Series(data=pred_ids.values).rename('domain')
                            .to_frame()
                            .join(domain.areas, on='domain')
                            .drop('domain', axis=1))
                    for area in area_names:
                        pred_ids_area = pred_ids_areas[area]
                        if pred_ids_area.isnull().all():
                            top_area = np.nan
                        else:
                            top_area = pred_ids_area.value_counts().index[0]
                        pred_areas.set_value(id_, area, top_area)
                        nearest = near_areas[area].loc[id_]
                        is_pred_over_near_area = pred_ids_area.isin([nearest]).any()
                        over_areas.set_value(id_, area, is_pred_over_near_area)
                print(best_pred.to_frame().mean(axis=0))
                print(over_areas.mean(axis=0))
                
                sp_pred_dict = {}
                for id_, sp_pred in sp_preds.items():
                    coords = sp_pred['most_likely'].coords
                    sp_pred_dict[id_] = {
                            'lat': coords['lat'].iloc[0],
                            'lon': coords['lon'].iloc[0]
                            }
                sp_pred = pd.DataFrame.from_dict(sp_pred_dict, orient='index')
                # Aggregate model results and add to fold_results DataFrame
                model_s = pd.Series(model_name, index=sp_test.ids, name='model')
                model_results = pd.concat([seeds_s, model_s, fold_s, errors, 
                    coverage, best_pred,
                    sp_pred.add_prefix(pred_prefix),
                    sp_test.coords.add_prefix(true_prefix),
                    pred_areas.add_prefix(pred_prefix),
                    near_areas.add_prefix(near_prefix),
                    true_areas.add_prefix(true_prefix),
                    over_areas.add_prefix(over_prefix)], axis=1)
                fold_results = pd.concat([fold_results, model_results], axis=0)
        
        fold_results = fold_results[col_names]
        fold_results.index.name = 'ids'
        results = pd.concat([results, fold_results], axis=0)
    
    results = results[col_names]  # order columns as in col_names
    results.index.name = 'ids'
    results.to_csv(output)


def weight_points_by_population(sp, population, area):
    relative_population = population / population.sum()
    weight_by_area = relative_population.div(sp.areas[area].value_counts()).rename('weight')
    weights = (sp.areas[area].to_frame()
            .join(weight_by_area, on=area)
            .drop(area, axis=1)
            .squeeze())
    return weights

if __name__ == '__main__':
    cross_validate()

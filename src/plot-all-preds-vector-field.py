import click

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

from models.points import SpatialPoints
from models.bda import GaussianKernelSmoother

@click.command()
@click.option('--title', default=None)
@click.option('--model', default=None)
@click.option('--bandwidth', default=100)
@click.option('--domain-fp')
@click.option('--model-name', default='')
@click.option('--map-polygon')
@click.option('--width', default=10)
@click.option('--height', default=10)
@click.argument('input_')
@click.argument('output')
def plot_all_preds(input_, output, title, model, model_name, map_polygon, width,
    height, domain_fp, bandwidth):
    results = pd.read_csv(input_)
    if model:
        results = results[results['model'] == model]
    true_coords = results[['ids', 'true_lat', 'true_lon']]
    true_coords.columns = ['ids', 'lat', 'lon']
    #true_coords['type'] = pd.Series('true', index=true_coords.index)
    pred_coords = results[['ids', 'pred_lat', 'pred_lon']]
    pred_coords.columns = ['ids', 'lat', 'lon']
    #pred_coords['type'] = pd.Series('pred', index=pred_coords.index)

    true_coords.set_index('ids', inplace=True)
    sp_true = SpatialPoints(coords=true_coords)
    pred_coords.set_index('ids', inplace=True)

    domain = pd.read_csv(domain_fp, index_col='domain')
    #domain = domain.iloc[::2, :]
    domain = SpatialPoints(coords=domain[['lat', 'lon']])
    smoother = GaussianKernelSmoother(bandwidth)
    weights = smoother.compute_weights(domain, sp_true)
    weights = weights.unstack(level=-1).as_matrix()
    preds = pred_coords.sort_index().as_matrix()
    preds = np.dot(weights, preds) / np.sum(weights, axis=1)[:, np.newaxis]
    preds = pd.DataFrame(data=preds, index=domain.ids, columns=['lat', 'lon'])
    sp_pred = SpatialPoints(coords=preds.copy())
    preds.rename(columns={'lat': 'pred_lat', 'lon': 'pred_lon'}, inplace=True)
    coords = pd.concat([domain.coords, preds], axis=1)
    coords['delta_lon'] = coords['pred_lon'] - coords['lon']
    coords['delta_lat'] = coords['pred_lat'] - coords['lat']
    coords['length'] = np.sqrt(np.square(coords['delta_lon']) + np.square(coords['delta_lat']))
    coords['scale'] = 0.02 / coords['length']

    error = domain.distances_from(sp_pred)
    bins = [0., 25, 50, 100, 200, 300, 400, 500, 1000, 3000]
    bin_labels = ['<' + str(b) for b in bins[1:-1]] + [str(bins[-2]) + '+']
    error_categorical = pd.cut(error, bins, labels = bin_labels, include_lowest=True).rename('error_categorical')
    coords = pd.concat([coords, error_categorical], axis=1)


    #ro.globalenv['n_total'] = all_coords.shape[0] / 2
    ro.globalenv['bin_labels'] = bin_labels
    ro.globalenv['model_name'] = model_name
    ro.globalenv['map_polygon'] = map_polygon
    ro.globalenv['width'] = width
    ro.globalenv['height'] = height
    ro.globalenv['output'] = output

    ro.globalenv['coords'] = pandas2ri.py2ri(coords)

    r("""
    library(ggplot2)
    library(viridis)
    library(gridExtra)
    library(grid)
    library(maps)

    if (map_polygon == 'triangle') {
        map <- map_data('county')
        map <- subset(map, region == 'north carolina')
        map <- subset(map, subregion %in% c('wake', 'orange', 'durham'))
    } else {
        map <- map_data(map_polygon)
    }

    theme_set(theme_minimal() + theme(line = element_blank(),
        axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank()))

    coords$error_categorical <- factor(coords$error_categorical, levels = bin_labels)
    p <- ggplot(coords, aes(lon - scale * delta_lon / 2, lat - scale * delta_lat / 2, color = error_categorical))
    p <- p + geom_polygon(data = map, aes(x = long, y = lat, group = group), fill = NA, color = 'black')
    p <- p + geom_segment(aes(xend = lon + scale * delta_lon / 2, yend = lat + scale * delta_lat / 2),
        arrow = arrow(length = unit(0.1, "cm")))
    p <- p + scale_color_viridis(discrete = TRUE, direction = -1)
    p <- p + theme(legend.position = 'top')
    p <- p + guides(color = guide_legend(title = 'Average Error (km)', nrow = 1))
    p <- p + coord_fixed(1.3)
    p
    ggsave(output, units='in', width=width, height=height)
    #png(output, units='in', res=200, width=width, height=height)
    #grid_arrange_shared_legend(plots[4:1], nrow = 2, ncol = 2, position = 'top')
    #dev.off()
    """)

if __name__ == '__main__':
    plot_all_preds()

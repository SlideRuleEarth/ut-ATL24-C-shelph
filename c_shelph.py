import numpy as np
import datetime
import traceback
import pandas as pd
import copy


def bin_data(dataset, lat_res, height_res):
    '''Bin data along vertical and horizontal scales for later segmentation'''
    
    # Calculate number of bins required both vertically and horizontally with resolution size
    lat_bin_number = round(abs(dataset['latitude'].min() - dataset['latitude'].max())/lat_res)

    height_bin_number = round(abs(dataset['photon_height'].min() - dataset['photon_height'].max())/height_res)
    
     # Duplicate dataframe
    dataset1 = dataset
    
    # Cut lat bins
    lat_bins = pd.cut(dataset['latitude'], lat_bin_number, labels = np.array(range(lat_bin_number)))
    
    # Add bins to dataframe
    dataset1['lat_bins'] = lat_bins
    
    pd.options.mode.chained_assignment = None 
    # Cut height bins
    height_bins = pd.cut(dataset['photon_height'], height_bin_number, labels = np.round(np.linspace(dataset['photon_height'].min(), dataset['photon_height'].max(), num=height_bin_number), decimals = 1))
    
    pd.options.mode.chained_assignment = 'warn' 
    # Add height bins to dataframe
    dataset1['height_bins'] = height_bins
    dataset1 = dataset1.reset_index(drop=True)

    return dataset1


def get_sea_height(binned_data, surface_buffer=-0.5):
    '''Calculate mean sea height for easier calculation of depth and cleaner figures'''
    
    # Create sea height list
    sea_height = []
    
    # Group data by latitude
    binned_data_sea = binned_data[(binned_data['photon_height'] > surface_buffer)] # Filter out subsurface data
    grouped_data = binned_data_sea.groupby(['lat_bins'], group_keys=True)
    data_groups = dict(list(grouped_data))
    
    # Loop through groups and return average sea height
    for k,v in data_groups.items():
        # Create new dataframe based on occurance of photons per height bin
        new_df = pd.DataFrame(v.groupby('height_bins').count())

        if not new_df.empty:
            
            # Return the bin with the highest count
            largest_h_bin = new_df['latitude'].argmax()
            
            # Select the index of the bin with the highest count
            largest_h = new_df.index[largest_h_bin]
            
            # Calculate the median value of all values within this bin
            lat_bin_sea_median = v.loc[v['height_bins']==largest_h, 'photon_height'].median()
            
            # Append to sea height list
            sea_height.append(lat_bin_sea_median)
            del new_df
            
    # Filter out sea height bin values outside 2 SD of mean.
    mean = np.nanmean(sea_height, axis=0)
    sd = np.nanstd(sea_height, axis=0)
    sea_height_1 = np.where((sea_height > (mean + 2*sd)) | (sea_height < (mean - 2*sd)), np.nan, sea_height).tolist()
    
    return sea_height_1


def get_bath_height(binned_data, percentile, WSHeight, height_resolution):
    """
        Calculates the bathymetry level (depth) for each bin in a 2D grid based on 
        photon counts and a specified percentile threshold.

    """
    # Create sea height list
    bath_height = []
    
    geo_index_ph = []
    geo_temp_ind = []
    geo_photon_height = []
    geo_longitude = []
    geo_latitude = []
    
    # Group data by latitude
    # Filter out surface data that are two bins below median surface value calculated above
    binned_data_bath = binned_data[(binned_data['photon_height'] < WSHeight - (height_resolution * 2))]
    grouped_data = binned_data_bath.groupby(['lat_bins'], group_keys=True)
    data_groups = dict(list(grouped_data))

    # Create a percentile threshold of photon counts in each grid, grouped by both x and y axes.
    count_threshold = np.percentile(binned_data.groupby(['lat_bins', 'height_bins']).size().reset_index().groupby('lat_bins')[[0]].max(), percentile)
    
    counts_in_bins = []
    # Loop through groups and return average bathy height
    for k,v in data_groups.items():
        new_df = pd.DataFrame(v.groupby('height_bins').count())
        
        if not new_df.empty:

            bath_bin = new_df['latitude'].argmax()
            bath_bin_h = new_df.index[bath_bin]

            counts_in_bins.append(new_df.iloc[bath_bin]['latitude'])

    counts_in_bins = np.asarray(counts_in_bins)
    cib_thresh_85 = np.percentile(counts_in_bins, 85)
    cib_thresh_65 = np.percentile(counts_in_bins, 65)

    if cib_thresh_85 == cib_thresh_65:

        print('Likely No bathymetry, normal distribution of photons.')
        if cib_thresh_85 <= 5:
            print('C-Shelph too few photons per bin. Setting min photons to 6.')
            counts_in_bins_thresh = 6
        else:
            counts_in_bins_thresh = cib_thresh_85

    else: 
        if cib_thresh_65 <= 5:
            print('C-Shelph too few photons per bin. Setting min photons to 6.')
            counts_in_bins_thresh = 6
        else:
            counts_in_bins_thresh = cib_thresh_65
            print('C-Shelph, using lower thresh.')

    # Loop through groups and return average bathy height
    for k,v in data_groups.items():
        new_df = pd.DataFrame(v.groupby('height_bins').count())

        if not new_df.empty:
            # print('new_df: ', new_df)
            bath_bin = new_df['latitude'].argmax()
            bath_bin_h = new_df.index[bath_bin]
            
            # Set threshold of photon counts per bin
            if new_df.iloc[bath_bin]['latitude'] >= counts_in_bins_thresh:
                
                geo_photon_height.append(v.loc[v['height_bins']==bath_bin_h, 'photon_height'].values)
                geo_longitude.append(v.loc[v['height_bins']==bath_bin_h, 'longitude'].values)
                geo_latitude.append(v.loc[v['height_bins']==bath_bin_h, 'latitude'].values)
                geo_index_ph.append(v.loc[v['height_bins']==bath_bin_h, 'index_ph'].values)
                geo_temp_ind.append(v.loc[v['height_bins']==bath_bin_h, 'temp_index'].values)
                
                bath_bin_median = v.loc[v['height_bins']==bath_bin_h, 'photon_height'].median()
                bath_height.append(bath_bin_median)
                del new_df
                
            else:
                bath_height.append(np.nan)
                del new_df

    try:
        geo_index_ph_list = np.concatenate(geo_index_ph).ravel().tolist()
        geo_temp_ind_list = np.concatenate(geo_temp_ind).ravel().tolist()
        geo_longitude_list = np.concatenate(geo_longitude).ravel().tolist()
        geo_latitude_list = np.concatenate(geo_latitude).ravel().tolist()
        geo_photon_list = np.concatenate(geo_photon_height).ravel().tolist()
        geo_depth = WSHeight - geo_photon_list
        geo_df = pd.DataFrame({'index_ph': geo_index_ph_list, 'PC_index': geo_temp_ind_list,'longitude': geo_longitude_list,
                            'latitude':geo_latitude_list, 'photon_height': geo_photon_list, 'depth':geo_depth})
    
        del geo_longitude_list, geo_latitude_list, geo_photon_list

        return bath_height, geo_df

    except Exception as c_shelph_err:

        print('c_shelph_err: ', c_shelph_err)
    
        return None, None


def c_shelph_classification(point_cloud, surface_buffer=-0.5, h_res=0.5, lat_res=0.001,
                            thresh=20, min_buffer=-80, max_buffer=5,
                            sea_surface_label=None, bathymetry_label=None):
    
    class_arr = point_cloud['class_ph'].to_numpy()
    sea_surface_indices = np.argwhere(class_arr == sea_surface_label).flatten()

    # Aggregate data into dataframe
    dataset_sea = pd.DataFrame({'index_ph': point_cloud['index_ph'].values,
                                'temp_index': np.arange(0, (point_cloud.shape[0]), 1),
                                'latitude': point_cloud['latitude'].values,
                                'longitude': point_cloud['longitude'].values,
                                'photon_height': point_cloud['geoid_corr_h']},
                           columns=['index_ph', 'temp_index', 'latitude', 'longitude', 'photon_height'])
    
    # dataset_sea = dataset_sea[(point_cloud.heights['signal_conf_ph_1'].values != 0)  & (point_cloud.heights['signal_conf_ph_1'].values != 1)]
    # dataset_sea1 = dataset_sea[(point_cloud.heights['signal_conf_ph_3'].values != 0)  & (point_cloud.heights['signal_conf_ph_4'].values != 0)]

    # Filter for elevation range
    dataset_sea1 = dataset_sea[(dataset_sea['photon_height'] > min_buffer) & (dataset_sea['photon_height'] < max_buffer)]
    
    binned_data_sea = bin_data(dataset_sea1, lat_res, h_res)
    binned_data_sea["height_bins"] = pd.to_numeric(binned_data_sea["height_bins"])

    # Find mean sea height, Currently not used, but will add later.
    sea_height = get_sea_height(binned_data_sea, surface_buffer)

    med_water_surface = np.nanmedian(point_cloud['geoid_corr_h'].to_numpy()[sea_surface_indices])

    bath_height, geo_df = get_bath_height(binned_data_sea, thresh, med_water_surface, h_res)

    if geo_df is not None:

        # Remove Bathy points without seasurface above.
        # sea_surf_lats = dataset_sea['latitude'][sea_surface_indices]
        # bathy_keep = _array_for_loop(geo_df['latitude'].to_numpy(), surf_lats=sea_surf_lats)
        # geo_df = geo_df[bathy_keep]

        classifications = np.zeros((point_cloud.shape[0]))
        classifications[:] = 0
        
        classifications[geo_df['PC_index'].to_numpy()] = bathymetry_label  # sea floor

        unique_bathy_filterlow = np.argwhere(point_cloud['geoid_corr_h'] > (med_water_surface - 1.75)).flatten()
        
        classifications[geo_df['PC_index'].to_numpy()] = bathymetry_label
        classifications[unique_bathy_filterlow] = 0
        classifications[sea_surface_indices] = sea_surface_label  # sea surface

        results = {'classification': classifications}
        
        return results
    else:

        classifications = np.zeros((point_cloud.shape[0]))
        classifications[:] = 0
        classifications[sea_surface_indices] = sea_surface_label  # sea surface
        results = {'classification': classifications}

        return results


def slice_binned_data_by_height(bin_data=None, med_water_surface=None, high=None, low=None):

        sliced_bin_data = bin_data.loc[(bin_data['height_bins'] <= high) & (bin_data['height_bins'] > low)]

        return sliced_bin_data


def generate_along_track_height_bins(binned_data=None, med_water_surface=None, surface_buffer=None, max_depth=None, height_steps=None):

    med_water_surface = med_water_surface - surface_buffer

    vertical_bin_groups = create_bin_groups(strt_ind=med_water_surface, end_ind=med_water_surface + max_depth, step=height_steps, iterate=False)

    if len(vertical_bin_groups) > 0:

        for group in vertical_bin_groups:

            sliced_height_bins = slice_binned_data_by_height(bin_data=copy.deepcopy(binned_data),
                                                                med_water_surface=med_water_surface,
                                                                high=group[0], low=group[1])

            yield sliced_height_bins

    else:
        return [None]
    



# def remove_bathy_without_surface(x, surf_lats):

#     if abs(x - surf_lats).min() < 0.005:
#         return True
#     else:
#         return False


# def _array_for_loop(x, surf_lats=None):
#     return np.array([remove_bathy_without_surface(xi, surf_lats) for xi in x])


def create_bin_groups(strt_ind=None, end_ind=None, step=None, iterate=True):

    try: 
        start_end_bins = []

        if strt_ind == end_ind:
            return start_end_bins
        
        if np.isnan(strt_ind):
            return start_end_bins
        
        if np.isnan(end_ind):
            return start_end_bins


        lat_bin_groups = np.arange(strt_ind, end_ind, step)
        if end_ind % step > 0:
            lat_bin_groups = np.concatenate((lat_bin_groups, np.array([end_ind])))

        if iterate:
            iterate = 1
        else:
            iterate = 0

        for ind, i in enumerate(lat_bin_groups):

            if ind == lat_bin_groups.shape[0]-2:
                start_end_bins.append((i+iterate, lat_bin_groups[ind+1]))
                break
                
            elif ind > 0:
                start_end_bins.append((i+iterate, lat_bin_groups[ind+1]))
        
            else:
                if lat_bin_groups.shape[0] > 1:
                    start_end_bins.append((iterate, lat_bin_groups[ind+1]))

        return start_end_bins
    
    except Exception as create_bins_err:

        print('create_bins_err: ', create_bins_err)
        print(str(traceback.format_exc()))

        return []



def plot_pointcloud(classified_pointcloud=None, output_path=None):

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    ylim_min = -80
    ylim_max = 20

    xlim_min = 24.5
    xlim_max = 25

    plt.figure(figsize=(48, 16))
    
    plt.plot(classified_pointcloud['latitude'][classified_pointcloud['classifications'] == 0.0],
                classified_pointcloud['geoid_corr_h'][classified_pointcloud['classifications'] == 0.0],
                'o', color='0.7', label='Other', markersize=2, zorder=1)
    
    plt.plot(classified_pointcloud['latitude'][classified_pointcloud['classifications'] == 41.0],
                classified_pointcloud['geoid_corr_h'][classified_pointcloud['classifications'] == 41.0],
                'o', color='blue', label='Other', markersize=5, zorder=5)
    
    plt.plot(classified_pointcloud['latitude'][classified_pointcloud['classifications'] == 40.0],
                classified_pointcloud['geoid_corr_h'][classified_pointcloud['classifications'] == 40.0],
                'o', color='red', label='Other', markersize=5, zorder=5)

#         plt.scatter(point_cloud.x[point_cloud._bathy_classification_counts == 1],
#                  point_cloud.z[point_cloud._bathy_classification_counts == 1],
#                  s=1, marker='.', c=point_cloud._bathy_classification_counts[point_cloud._bathy_classification_counts == 1], cmap='cool', vmin=0, vmax=1, label='Seabed')
    # if point_cloud._z_refract is not None:
    #     if point_cloud._z_refract.any():
    #         plt.scatter(point_cloud.y[point_cloud._bathy_classification_counts > 0],
    #             point_cloud._z_refract[point_cloud._bathy_classification_counts > 0],
    #             s=36, marker='o', c=point_cloud._bathy_classification_counts[point_cloud._bathy_classification_counts > 0], cmap='Reds', vmin=0, vmax=1, label='Refraction Corrected', zorder=11)

    plt.xlabel('Latitude (degrees)', fontsize=36)
    plt.xticks(fontsize=34)
    plt.ylabel('Height (m)', fontsize=36)
    plt.yticks(fontsize=34)
    plt.ylim(ylim_min, ylim_max)
    # plt.xlim(xlim_min, xlim_max)
    plt.title('Final Classifications - ', fontsize=40)
    # plt.title(fname + ' ' + channel)
    plt.legend(fontsize=36)
    
    plt.savefig(output_path)
    plt.close()


    return



def main(args):

    input_fname = args.beam_data_csv
    output_label_fname = args.output_data_csv

    sea_surface_label = 41
    bathymetry_label = 40

    point_cloud = pd.read_csv(input_fname)

    # Start Bathymetry Classification

    c_shelph_results = c_shelph_classification(copy.deepcopy(point_cloud), surface_buffer=-0.5,
                                                                        h_res=0.5, lat_res=0.001, thresh=0.5,
                                                                        min_buffer=-80, max_buffer=5,
                                                                        # sea_surface_indices=sea_surface_inds,
                                                                        sea_surface_label=sea_surface_label,
                                                                        bathymetry_label=bathymetry_label)

    point_cloud['classifications'] = c_shelph_results['classification']


    plot_path = output_label_fname.replace('.csv', '.png')
    # plot_pointcloud(classified_pointcloud=point_cloud, output_path=plot_path)

    point_cloud.to_csv(output_label_fname)

    return


if __name__=="__main__":

    import argparse
    import sys
    import numpy as np
    import traceback
    import pandas as pd

    parser = argparse.ArgumentParser()

    # <configuration json> <beam information json> <beam data csv> <output data csv>

    parser.add_argument("--configuration-json")
    parser.add_argument("--beam-information-json")
    parser.add_argument("--beam-data-csv")
    parser.add_argument("--output-data-csv")

    args = parser.parse_args()

    main(args)

    sys.exit(0)

    # python3 /home/mjh5468/local_repo_development/ATL24_C-shelph/c_shelph.py --configuration-json '' --beam-information-json '' --beam-data-csv '/home/mjh5468/test_data/SLIDERULE_testing/bathy_spot_3.csv' --output-data-csv '/home/mjh5468/test_data/SLIDERULE_testing/bathy_spot_3_classified.csv'





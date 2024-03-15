import numpy as np
import datetime
import traceback
import pandas as pd
import copy
import scipy


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
    
    geo_ph_index = []
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
                geo_ph_index.append(v.loc[v['height_bins']==bath_bin_h, 'ph_index'].values)
                geo_temp_ind.append(v.loc[v['height_bins']==bath_bin_h, 'temp_index'].values)
                
                bath_bin_median = v.loc[v['height_bins']==bath_bin_h, 'photon_height'].median()
                bath_height.append(bath_bin_median)
                del new_df
                
            else:
                bath_height.append(np.nan)
                del new_df

    try:
        geo_ph_index_list = np.concatenate(geo_ph_index).ravel().tolist()
        geo_temp_ind_list = np.concatenate(geo_temp_ind).ravel().tolist()
        geo_longitude_list = np.concatenate(geo_longitude).ravel().tolist()
        geo_latitude_list = np.concatenate(geo_latitude).ravel().tolist()
        geo_photon_list = np.concatenate(geo_photon_height).ravel().tolist()
        geo_depth = WSHeight - geo_photon_list
        geo_df = pd.DataFrame({'ph_index': geo_ph_index_list, 'PC_index': geo_temp_ind_list,'longitude': geo_longitude_list,
                            'latitude':geo_latitude_list, 'photon_height': geo_photon_list, 'depth':geo_depth})
    
        del geo_longitude_list, geo_latitude_list, geo_photon_list

        return bath_height, geo_df

    except Exception as c_shelph_err:

        print('c_shelph_err: ', c_shelph_err)
    
        return None, None


def c_shelph_classification(point_cloud, sea_surface_indices=None,
                            surface_buffer=-0.5, h_res=0.5, lat_res=0.001,
                            thresh=20, min_buffer=-80, max_buffer=5,
                            sea_surface_label=None, bathymetry_label=None):

    # Aggregate data into dataframe
    dataset_sea = pd.DataFrame({'ph_index': point_cloud['ph_index'].values,
                                'temp_index': np.arange(0, (point_cloud.shape[0]), 1),
                                'latitude': point_cloud['lat_ph'].values,
                                'longitude': point_cloud['lon_ph'].values,
                                'photon_height': point_cloud['geoid_corrected_h']},
                           columns=['ph_index', 'temp_index', 'latitude', 'longitude', 'photon_height'])
    
    # dataset_sea = dataset_sea[(point_cloud.heights['signal_conf_ph_1'].values != 0)  & (point_cloud.heights['signal_conf_ph_1'].values != 1)]
    # dataset_sea1 = dataset_sea[(point_cloud.heights['signal_conf_ph_3'].values != 0)  & (point_cloud.heights['signal_conf_ph_4'].values != 0)]

    # Filter for elevation range
    dataset_sea1 = dataset_sea[(dataset_sea['photon_height'] > min_buffer) & (dataset_sea['photon_height'] < max_buffer)]
    
    binned_data_sea = bin_data(dataset_sea1, lat_res, h_res)
    binned_data_sea["height_bins"] = pd.to_numeric(binned_data_sea["height_bins"])

    # Find mean sea height
    sea_height = get_sea_height(binned_data_sea, surface_buffer)

    # Set sea height
    med_water_surface_h = np.nanmedian(sea_height)

    med_water_surface_h2 = np.nanmedian(point_cloud['geoid_corrected_h'].to_numpy()[sea_surface_indices])

    # all_sub_surface_bins = binned_data_sea.loc[binned_data_sea['height_bins'] < med_water_surface_h]

    bath_height, geo_df = get_bath_height(binned_data_sea, thresh, med_water_surface_h2, h_res)

    if geo_df is not None:

        # Remove Bathy points without seasurface above.
        sea_surf_lats = dataset_sea['latitude'][sea_surface_indices]
        # bathy_keep = _array_for_loop(geo_df['latitude'].to_numpy(), surf_lats=sea_surf_lats)
        # geo_df = geo_df[bathy_keep]

        classifications = np.zeros((point_cloud['h_ph'].to_numpy().shape))
        classifications[:] = 0
        
        classifications[geo_df['PC_index'].to_numpy()] = bathymetry_label  # sea floor

        unique_bathy_filterlow = np.argwhere(point_cloud['h_ph'] > (med_water_surface_h2 - 1)).flatten()
        # classifications[unique_bathy_filterlow] = 0
        classifications[sea_surface_indices] = sea_surface_label  # sea surface
        classifications[geo_df['PC_index'].to_numpy()] = bathymetry_label
        results = {'classification': classifications}
        
        return results
    else:

        classifications = np.zeros((point_cloud['h_ph'].to_numpy().shape))
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
    



def remove_bathy_without_surface(x, surf_lats):

    if abs(x - surf_lats).min() < 0.005:
        return True
    else:
        return False


def _array_for_loop(x, surf_lats=None):
    return np.array([remove_bathy_without_surface(xi, surf_lats) for xi in x])


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







def time2UTC(gps_seconds_array=None):

    # Number of Leap seconds
    # See: 'https://www.ietf.org/timezones/data/leap-seconds.list'
    # 15 - 1 Jan 2009
    # 16 - 1 Jul 2012
    # 17 - 1 Jul 2015
    # 18 - 1 Jan 2017
    leap_seconds = 18

    gps_start = datetime.datetime(year=1980, month=1, day=6)
    time_ph = [datetime.timedelta(seconds=time) for time in gps_seconds_array]
    # last_photon = datetime.timedelta(seconds=gps_seconds[-1])
    error = datetime.timedelta(seconds=leap_seconds)
    ph_time_utc = [(gps_start + time - error) for time in time_ph]

    return ph_time_utc










##################################
##################################
##################################
##################################
##################################

##################################











def binned_processing(pointcloud=None, window_size=None, sea_surface_label=None):
    """
        Process a given profile to produce a model.

        Args:
            profile: The profile object to be processed.

        Returns:
            Model: A Model object containing the processed data.
    """
    step_along_track = 1

    range_z = (-100, 100)
    res_z = 0.5
    res_along_track = 800 #100

    z_min = range_z[0]
    z_max = range_z[1]
    z_bin_count = np.int64(np.ceil((z_max - z_min) / res_z))
    bin_edges_z = np.linspace(z_min, z_max, num=z_bin_count+1)

    data = copy.deepcopy(pointcloud)

    # along track bin sizing
    #   get the max index of the dataframe
    #   create bin group ids (at_idx_grp) based on pho_count spacing
    at_max_idx = data.x_ph.max()
    at_min_idx = data.x_ph.min()
    at_idx_grp = np.arange(at_min_idx, at_max_idx + res_along_track, res_along_track)
    
    # sort the data by distnace along track, reset the index
    # add 'at_grp' column for the bin group id
    data.sort_values(by='x_ph', inplace=True)
    #data.reset_index(inplace=True)
    data['idx'] = data.index
    data['at_grp'] = 0

    # for each bin group, assign an interger id for index values between each of the 
    #   group bin values. is pho_count = 20 then at_idx_grp = [0,20,49,60...]
    #   - data indicies between 0-20: at_grp = 1
    #   - data indicies between 20-40: at_grp = 2...
    for i, grp in enumerate(at_idx_grp):
        if grp < at_idx_grp.max():
            data['at_grp'][data['x_ph'].between(at_idx_grp[i], at_idx_grp[i+1])] = (at_idx_grp[i] - at_idx_grp.min()) / res_along_track
    
    # add group bin columns to the profile, photon group bin index
    data['pho_grp_idx'] = data['at_grp']
    
    # calculating the range so the histogram output maintains exact along track values
    at_min = data.x_ph.min()
    xh = (data.x_ph.values)
    bin_edges_at_min = data.groupby('at_grp').x_ph.min().values
    bin_edges_at_max = data.groupby('at_grp').x_ph.max().values

    bin_edges_at = np.concatenate([np.array([data.x_ph.min()]), bin_edges_at_max])

    # array to store actual interpolated model and fitted model
    hist_modeled = np.nan * np.zeros((bin_edges_at.shape[0] -1, z_bin_count))
    
    start_step = (window_size) / 2
    end_step = len(bin_edges_at)

    # -1 to start index at 0 instead of 1. For correct indexing when writing to hist_modeled array.
    win_centers = np.arange(np.int64(start_step), np.int64(
        end_step), step_along_track) -1

    window_args = create_window_processing_args(data=data, window_size=window_size, win_centers=win_centers, bin_edges_at=bin_edges_at)
    

    print('Starting Parallel Processing')
    parallel = True
    if parallel:

        processed_data = []
        with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
            processed_data = pool.starmap(process_window,
                                          zip(repeat(window_size),
                                              window_args['window_centres'],
                                                    repeat(z_min), repeat(z_max),
                                                    repeat(z_bin_count), repeat(res_z),
                                                    window_args['window_profiles'],
                                                    repeat(bin_edges_z), window_args['at_begs'],
                                                    window_args['at_ends']))
    
        replace_indices = np.hstack([elem[0] for elem in processed_data if elem[0] is not None])
        sea_surface = np.hstack([np.full(elem[0].shape[0], elem[1]) for elem in processed_data if elem[0] is not None])

        # print('replace_indices: ',replace_indices)
        # print('sea_surface: ',sea_surface)

        # sea_surface_values = np.zeros(pointcloud['ph_index'].to_numpy().shape[0])
        sea_surface_classifications = np.full(pointcloud['ph_index'].to_numpy().shape[0], sea_surface_label)
        classifications = np.zeros(pointcloud['ph_index'].to_numpy().shape[0])

        np.put(classifications, replace_indices, sea_surface_classifications)

        # np.put(sea_surface_values, replace_indices, sea_surface_label)
        pointcloud['classifications'] = classifications

        # for replace_index, ssurf in zip(replace_indices, sea_surface):

        #     pointcloud.loc[pointcloud['ph_index'] == replace_index] = ssurf
            # update_DF(df=pointcloud, col_ind_val=replace_indices, update_val=sea_surface)

            # def update_DF(df=None, col_ind_val=None, update_val=None):
            #     df.loc[df['ph_index'] == col_ind_val] = update_val
            #     return df

    #     replace_with = np.vstack([np.flip(elem[1]) for elem in processed_data])

    #     np.put_along_axis(hist_modeled, replace_indices, replace_with, axis=0)\
        
    return pointcloud




def create_window_processing_args(data=None, window_size=None, win_centers=None, bin_edges_at=None):


    at_begs = []
    at_ends = []
    window_profiles = []
    window_centres = []

    xh = (data.x_ph.values)

    # if photon_bins == False: 

    for window_centre in win_centers:

        # get indices/at distance of evaluation window
        i_beg = np.int64(max((window_centre - (window_size-1) / 2), 0))
        i_end = np.int64(min((window_centre + (window_size-1) / 2), len(bin_edges_at)-2)) + 1
        # + 1 above pushes i_end to include up to the edge of the bin when used as index

        at_beg = bin_edges_at[i_beg]
        at_end = bin_edges_at[i_end]

        # could be sped up with numba if this is an issue
        # subset data using along track distance window
        i_cond = ((xh > at_beg) & (xh < at_end))

        # copy profile to avoid overwriting
        w_profile = copy.deepcopy(data)

        w_profile = w_profile.loc[i_cond, :]

        # # remove all data except for the photons in this window
        # w_profile = df_win

        at_begs.append(at_beg)
        at_ends.append(at_end)
        window_profiles.append(w_profile)
        window_centres.append(window_centre)
  
    return {'at_begs': at_begs,
            'at_ends': at_ends,
            'window_profiles': window_profiles,
            'window_centres': window_centres}

def process_window(window_size=None, window_centre=None,
                   z_min=None, z_max=None, z_bin_count=None,
                   res_z=None,
                   win_profile=None, bin_edges_z=None,
                   at_beg=None, at_end=None):

    # version of dataframe with only nominal photons
    # use this data for constructing waveforms
    df_win_nom = win_profile.loc[win_profile.quality_ph == 0]

    height_geoid = df_win_nom.geoid_corrected_h.values

    # print()

    # subset of histogram data in the evaluation window
    h_ = histogram1d(height_geoid,
                        range=[z_min, z_max], bins=z_bin_count)

    # smooth the histogram with 0.2 sigma gaussian kernel
    h = gaussian_filter1d(h_, 0.2/res_z)

    # identify median lat lon values of photon data in this chunk
    # x_win = df_win_nom.lon_ph.median()
    # y_win = df_win_nom.lat_ph.median()
    # at_win = df_win_nom.x_ph.median()
    # any_sat_ph = (win_profile.quality_ph > 0).any()

    photon_inds, sea_surf = find_sea_surface(hist=np.flip(h), z_inv=-np.flip(bin_edges_z), win_profile=win_profile)

    return photon_inds, sea_surf

















def find_sea_surface(hist=None, z_inv=None, win_profile=None):

    peak_info = get_peak_info(hist, z_inv)

    z_bin_size = np.unique(np.diff(z_inv))[0]
    zero_val = 1e-31
    quality_flag = False
    params_out = {}
    # ########################## ESTIMATING SURFACE PEAK PARAMETERS #############################

    # Surface return - largest peak
    peak_info.sort_values(by='prominences', inplace=True, ascending=False)


    if peak_info.empty:
        return None, None

    # if the top two peaks are within 20% of each other
    # go with the higher elev one/the one closest to 0m
    # mitigating super shallow, bright reefs where seabed is actually brighter

    if peak_info.shape[0] > 1:

        # check if second peak is greater than 20% the prominence of the primary
        two_tall_peaks = (
            (peak_info.iloc[1].prominences) / peak_info.iloc[0].prominences) > 0.2

        if two_tall_peaks:
            # use the peak on top of the other
            # anywhere truly open water will have the surface as the top peak
            pks2 = peak_info.iloc[[0, 1]]
            peak_above = pks2.z_inv.argmin()
            surf_pk = peak_info.iloc[peak_above]

        else:
            surf_pk = peak_info.iloc[0]
    else:

        print('peak_info: ', peak_info)
        surf_pk = peak_info.iloc[0]

    # estimate noise rates above surface peak and top of surface peak

    # dont use ips, it will run to the edge of an array with a big peak
    # using std estimate of surface peak instead
    surface_peak_left_edge_i = np.int64(
        np.floor(surf_pk.i - 2 * surf_pk.sigma_est_left_i))

    # distance above surface peak to start estimating noise

    height_above_surf_noise_est = 30 # meters

    above_surface_noise_left_edge_i = surface_peak_left_edge_i - (height_above_surf_noise_est / z_bin_size)

    # # if theres not 15m above surface, use what there is above the surface
    if above_surface_noise_left_edge_i <= 0:
        above_surface_noise_left_edge_i = 0

    above_surface_idx_range = np.arange(above_surface_noise_left_edge_i, surface_peak_left_edge_i, dtype=np.int64)

    # # above surface estimation
    if surface_peak_left_edge_i <= 0:
        # no bins above the surface
        params_out['background'] = zero_val

    else:
        # median of all bins 15m above the peak
        params_out['background'] = np.median(
            hist[above_surface_idx_range]) + zero_val  # eps to avoid 0

    # use above surface data to refine the estimate of the top of the surface
    # top of the surface is the topmost surface bin with a value greater than the noise above
    
    # photons through the top of the surface
    top_to_surf_center_i = np.arange(surf_pk.i+1, dtype=np.int64)
    less_than_noise_bool = hist[top_to_surf_center_i] <= (2 * params_out['background'])
    # found just using the noise estimate x 1 leads to too wide of a window in quick testing, using x2 for now

    # get the lowest elevation bin higher than the above surface noise rate
    # set as the top of the surface

    if less_than_noise_bool.any():
        surface_peak_left_edge_i = np.where(less_than_noise_bool)[0][-1]
    
    # else defaults to 3 sigma range defined above

    # but this is a rough estimate of the surface peak assuming perfect gaussian on peak
    # surface peak is too important to leave this rough estimate
    # surf peak can have a turbid subsurface tail, too

    # elbow detection - where does the surface peak end and become subsurface noise/signal?
    # basically mean value theorem applied to the subsurface histogram slope
    # how to know when the histogram starts looking like possible turbidity?
    # we assume the surface should dissipate after say X meters depth, but will sooner in actuality
    # therefore, the actual numerical slope must cross the threshold required to dissipate after X meters at some point
    # where X is a function of surface peak width

    # what if the histogram has a rounded surface peak, or shallow bathy peaks?
    # rounded surface peak would be from pos to neg (excluded)
    # shallow bathy peaks would be the 2nd or 3rd slope threshold crossing
    # some sketches with the curve and mean slope help show the sense behind this

    # we'll use this to improve our estimate of the surface gaussian early on
    # elbow detection start
    dissipation_range = 10  # m # was originally 3m but not enough for high turbid cases
    slope_thresh = -surf_pk.heights / (dissipation_range/z_bin_size)
    diffed_subsurf = np.diff(hist[np.int64(surf_pk.i):])

    # detection of slope decreasing in severity, crossing the thresh
    sign_ = np.sign(diffed_subsurf - slope_thresh)
    # ie where sign changes (from negative to positive only)
    sign_changes_i = np.where(
        (sign_[:-1] != sign_[1:]) & (sign_[:-1] < 0))[0] + 1

    # if len(sign_changes_i) == 0:
    #     no_sign_change = True
    #     # basically aa surface at the bottom of window somehow
    #     quality_flag = -4
    #     # params_out = pd.Series(params_out, name='params', dtype=np.float64)
    #     bathy_quality_ratio = -1
    #     surface_gm = None 
    #     return params_out, quality_flag, bathy_quality_ratio, surface_gm

    # else:
    # calculate corner details
    transition_i = np.int64(surf_pk.i) + \
        sign_changes_i[0]  # full hist index

    # bottom bound of surface peak in indices
    surface_peak_right_edge_i = transition_i + 1
    # params_out['column_top'] = -z_inv[surface_peak_right_edge_i]

    # end elbow detection

    if surface_peak_right_edge_i > len(hist):
        surface_peak_right_edge_i = len(hist)


    surf_range_i = np.arange(surface_peak_left_edge_i,
                             surface_peak_right_edge_i, dtype=np.int64)
    
    # print('surface_peak_left_edge_i: ', -z_inv[surface_peak_left_edge_i])
    # print('surface_peak_right_edge_i: ', -z_inv[surface_peak_right_edge_i])

    clipped_prof = win_profile.loc[(win_profile['geoid_corrected_h'] < -z_inv[surface_peak_left_edge_i]) & (win_profile['geoid_corrected_h'] > -z_inv[surface_peak_right_edge_i])]
    geoid_corrected_h = win_profile.geoid_corrected_h
    z_surf = geoid_corrected_h[(geoid_corrected_h < -z_inv[surface_peak_left_edge_i]) & (geoid_corrected_h > -z_inv[surface_peak_right_edge_i])]
    z_inv_surf = -z_surf

    # print('z_surf: ', z_surf)
    # print('z_surf.median(): ', z_surf.median())

    # print('clipped_prof: ', clipped_prof)

    return clipped_prof['ph_index'].to_numpy(), z_surf.median()
    


def get_peak_info(hist, z_inv, verbose=False):
    """
    Evaluates input histogram to find peaks and associated peak statistics.
    
    Args:
        hist (array): Histogram of photons by z_inv.
        z_inv (array): Centers of z_inv bins used to histogram photon returns.
        verbose (bool, optional): Option to print output and warnings. Defaults to False.

    Returns:
        Pandas DataFrame with the following columns:
            - i: Peak indices in the input histogram.
            - prominences: Prominence of the detected peaks.
            - left_bases, right_bases: Indices of left and right bases of the peaks.
            - left_z, right_z: z_inv values of left and right bases of the peaks.
            - heights: Heights of the peaks.
            - fwhm: Full Width at Half Maximum of the peaks.
            - left_ips_hm, right_ips_hm: Left and right intersection points at half maximum.
            - widths_full: Full width of peaks.
            - left_ips_full, right_ips_full: Left and right intersection points at full width.
            - sigma_est_i: Estimated standard deviation indices.
            - sigma_est: Estimated standard deviation in units of z_inv.
            - prom_scaling_i, mag_scaling_i: Scaling factors for prominences and magnitudes using indices.
            - prom_scaling, mag_scaling: Scaling factors for prominences and magnitudes in units of z_inv.
            - z_inv: z_inv values at the peak indices.
    """
    
    z_inv_bin_size = np.unique(np.diff(z_inv))[0]

    # padding elevation mapping for peak finding at edges
    z_inv_padded = z_inv

    # left edge
    z_inv_padded = np.insert(z_inv_padded,
                             0,
                             z_inv[0] - z_inv_bin_size)  # use zbin for actual fcn

    # right edge
    z_inv_padded = np.insert(z_inv_padded,
                             len(z_inv_padded),
                             z_inv[-1] + z_inv_bin_size)  # use zbin for actual fcn

    dist_req_between_peaks = 0.49999  # m

    if dist_req_between_peaks/z_inv_bin_size < 1:
        warn_msg = '''get_peak_info(): Vertical bin resolution is greater than the req. min. distance 
        between peak. Setting req. min. distance = z_inv_bin_size. Results may not be as expected.
        '''
        if verbose:
            warnings.warn(warn_msg)
        dist_req_between_peaks = z_inv_bin_size

    # note: scipy doesnt identify peaks at the start or end of the array
    # so zeros are inserted on either end of the histogram and output indexes adjusted after

    # distance = distance required between peaks - use approx 0.5 m, accepts floats >=1
    # prominence = required peak prominence
    pk_i, pk_dict = find_peaks(np.pad(hist, 1),
                               distance=dist_req_between_peaks/z_inv_bin_size,
                               prominence=0.01)

    # evaluating widths with find_peaks() seems to be using it as a threshold - not desired
    # width = required peak width (index) - use 1 to return all
    # rel_height = how far down the peak to measure its width
    # 0.5 is half way down, 1 is measured at the base
    # approximate stdm from the full width and half maximum
    pk_dict['fwhm'], pk_dict['width_heights_hm'], pk_dict['left_ips_hm'], pk_dict['right_ips_hm'] \
        = peak_widths(np.pad(hist, 1), pk_i, rel_height=0.4)

    # calculate widths at full prominence, more useful than estimating peak width by std
    pk_dict['widths_full'], pk_dict['width_heights_full'], pk_dict['left_ips_full'], pk_dict['right_ips_full'] \
        = peak_widths(np.pad(hist, 1), pk_i, rel_height=1)

    # organize into dataframe for easy sorting and reindex
    pk_dict['i'] = pk_i - 1
    pk_dict['heights'] = hist[pk_dict['i']]

    # draw a horizontal line at the peak height until it cross the signal again
    # min values within that window identifies the bases
    # preference for closest of repeated minimum values
    # ie. can give weird values to the left/right of the main peak, and to the right of a bathy peak
    # when theres noise in a scene with one 0 bin somewhere far
    pk_dict['left_z'] = z_inv_padded[pk_dict['left_bases']]
    pk_dict['right_z'] = z_inv_padded[pk_dict['right_bases']]
    pk_dict['left_bases'] -= 1
    pk_dict['right_bases'] -= 1

    # left/right ips = interpolated positions of left and right intersection points
    # of a horizontal line at the respective evaluation height.
    # mapped to input indices so needs adjustingpk_dict['left_ips'] -= 1
    pk_dict['right_ips_hm'] -= 1
    pk_dict['left_ips_hm'] -= 1
    pk_dict['right_ips_full'] -= 1
    pk_dict['left_ips_full'] -= 1

    # estimate gaussian STD from the widths
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_i'] = (pk_dict['fwhm'] / 2.35)
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_left_i'] = (
        2*(pk_dict['i'] - pk_dict['left_ips_hm']) / 2.35)
    # sigma estimate in terms of int indexes
    pk_dict['sigma_est_right_i'] = (
        2*(pk_dict['right_ips_hm'] - pk_dict['i']) / 2.35)

    # sigma estimate in terms of int indexes
    pk_dict['sigma_est'] = z_inv_bin_size * (pk_dict['fwhm'] / 2.35)

    # approximate gaussian scaling factor based on prominence or magnitudes
    # for gaussians range indexed
    pk_dict['prom_scaling_i'] = pk_dict['prominences'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est_i'])
    pk_dict['mag_scaling_i'] = pk_dict['heights'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est_i'])

    # for gaussians mapped to z
    pk_dict['prom_scaling'] = pk_dict['prominences'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est'])
    pk_dict['mag_scaling'] = pk_dict['heights'] * \
        (np.sqrt(2 * np.pi) * pk_dict['sigma_est'])
    pk_dict['z_inv'] = z_inv[pk_dict['i']]

    peak_info = pd.DataFrame.from_dict(pk_dict, orient='columns')
    peak_info.sort_values(by='prominences', inplace=True, ascending=False)

    return peak_info



def first_pass_sea_surface(pointcloud=None, sea_surface_label=None):

    classified_pointcloud = binned_processing(pointcloud, window_size=3, sea_surface_label=sea_surface_label)

    return classified_pointcloud



def plot_pointcloud(classified_pointcloud=None, output_path=None):

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    ylim_min = -80
    ylim_max = 20

    xlim_min = 24.5
    xlim_max = 25

    plt.figure(figsize=(48, 16))
    
    plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 0.0],
                classified_pointcloud['geoid_corrected_h'][classified_pointcloud['classifications'] == 0.0],
                'o', color='0.7', label='Other', markersize=2, zorder=1)
    
    plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 41.0],
                classified_pointcloud['geoid_corrected_h'][classified_pointcloud['classifications'] == 41.0],
                'o', color='blue', label='Other', markersize=5, zorder=5)
    
    plt.plot(classified_pointcloud['lat_ph'][classified_pointcloud['classifications'] == 40.0],
                classified_pointcloud['geoid_corrected_h'][classified_pointcloud['classifications'] == 40.0],
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



if __name__=="__main__":

    import argparse
    import numpy as np
    import datetime
    import traceback
    import pandas as pd

    import multiprocessing
    from itertools import repeat
    import copy

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths
    from fast_histogram import histogram2d, histogram1d

    parser = argparse.ArgumentParser()

    parser.add_argument("--photon-data-fname", default=True)
    # parser.add_argument("--sea-surface-path", default=True)
    parser.add_argument("--output-label-fname", default=True)

    args = parser.parse_args()

    input_fname = args.photon_data_fname
    output_label_fname = args.output_label_fname

    sea_surface_label = 41
    bathymetry_label = 40

    point_cloud = pd.read_csv(input_fname)

    point_cloud['ph_index'] = np.arange(0, point_cloud.shape[0], 1)
    point_cloud['dist_ph_along_total'] = point_cloud['segment_dist_x'] + point_cloud['dist_ph_along']
    point_cloud['x_ph'] = point_cloud['dist_ph_along_total'] - point_cloud['dist_ph_along_total'].min()
    
    # calculate geodetic heights
    #   ellipsoidal height (heights/h_ph) - geoid (geophys/geoid)
    # point_cloud['z_ph'] = point_cloud['h_ph'] - point_cloud['geoid']
    # point_cloud['delta_time'] = time2UTC(gps_seconds_array=point_cloud['gps_seconds'].to_numpy())

    # print('point_cloud shape: ', point_cloud.shape)

    classified_pointcloud = first_pass_sea_surface(pointcloud=point_cloud,
                                                   sea_surface_label=sea_surface_label)
    
    class_arr = classified_pointcloud['classifications'].to_numpy()
    sea_surface_inds = np.argwhere(class_arr == sea_surface_label).flatten()

    # print('sea_surface_inds: ',sea_surface_inds)
    # print('sea_surface_inds shape: ',sea_surface_inds.shape)

    # /home/mjh5468/test_data/ATL24_medmodel_test_output/JW_ATL03_20210704212129_01721201_006_01_gt1r_labels.csv
    plot_path = output_label_fname.replace('.csv', '.png')

    # classified_pointcloud = classified_pointcloud.loc[(classified_pointcloud['lat_ph'] > 24.6) & (classified_pointcloud['lat_ph'] < 25)]
    bad_quality_inds = classified_pointcloud.loc[classified_pointcloud['quality_ph'] != 0]['ph_index'].to_numpy()

    plot_pointcloud(classified_pointcloud=classified_pointcloud, output_path=plot_path)

    # print('classified_pointcloud columns: ', classified_pointcloud.columns)
    # print('classified_pointcloud shape: ', classified_pointcloud.shape)

    c_shelph_results = c_shelph_classification(copy.deepcopy(classified_pointcloud), surface_buffer=-0.5,
                                                                        h_res=0.5, lat_res=0.001, thresh=0.5,
                                                                        min_buffer=-80, max_buffer=5,
                                                                        sea_surface_indices=sea_surface_inds,
                                                                        sea_surface_label=sea_surface_label,
                                                                        bathymetry_label=bathymetry_label)
    
    # print('c_shelph_results_results: ',c_shelph_results)

    c_shelph_results['classification'][bad_quality_inds] = 0

    classified_pointcloud['classifications'] = c_shelph_results['classification']

    plot_pointcloud(classified_pointcloud=classified_pointcloud, output_path=plot_path)
    
    classified_pointcloud[['classifications']].to_csv(output_label_fname)






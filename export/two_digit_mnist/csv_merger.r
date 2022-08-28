source('./_expand_name.r')

csv_merger = function(load_files_names, models_name_list) {
  combined_tables <- NULL
  # load tables for each element in the list
  tables <- lapply(load_files_names, read_csv)
  for (idx in 1:length(tables)) {
    t <- ldply(tables[idx], data.frame)  # convert from list to df
    # don't process dfs with no rows - to avoid dists where all configs failed to reach required max step
    if (!empty(t)) {
      # expand the name
      t <- expand.name(t)
      # rename model if names have been given
      if (length(models_name_list)) {
        t$model <- models_name_list[[idx]]      # rename the model name to pre-defined value in list
      }
      # only get common cols if a rbinded table exists (i.e. both dfs to merge do actually have cols)
      if (idx != 1) {
        common_cols <- intersect(colnames(combined_tables), colnames(t))  # get the common columns between the t tables to be merged
        combined_tables <- rbind(combined_tables[common_cols], t[common_cols])  # add model data to an accumulated table
      } else {
        combined_tables <- rbind(combined_tables, t)  # add model data to an accumulated table
      }
    }
  }
  return(combined_tables)
}

load_and_merge_csvs = function(lookup.name, single_filepath = NA) {
  csv_ext = '.csv'
  return(switch(
    lookup.name,
    "None" = csv_merger(
      list(single_filepath),
      list('Test')
    ),
###################################################################################################
    "div-1digit_conv-Adam-s3Init" = csv_merger(list(
      paste0(load_folder, 'ID79-1digit_conv-div', csv_ext),
      paste0(load_folder, 'ID84-1digit_conv-mlp_2L-256HU_wDecay-1e-3', csv_ext),
      paste0(load_folder, 'ID78-1digit_conv-nru_R30-40-2_gnc1_s3Init', csv_ext),
      paste0(load_folder, 'ID80-1digit_conv-realnpu-mod_R30-40-2_s3Init', csv_ext),
      paste0(load_folder, 'ID81-1digit_conv-nmruSign_R30-40-2_gncF_s3Init', csv_ext)
    ),
      list('DIV', 'MLP', 'NRU', 'Real NPU (mod)', 'NMRU')
    ),
###################################################################################################
    stop("Key given to csv_merger does not exist!")
  ))
}

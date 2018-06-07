# =====================================================================
# For compatibility with Rscript.exe: 
# =====================================================================
if(length(.libPaths()) == 1){
  # We're in Rscript.exe
  possible_lib_paths <- file.path(Sys.getenv(c('USERPROFILE','R_USER')),
                                  "R","win-library",
                                  paste(R.version$major,
                                        substr(R.version$minor,1,1),
                                        sep='.'))
  indx <- which(file.exists(possible_lib_paths))
  print(possible_lib_paths)
  if(length(indx)){
    .libPaths(possible_lib_paths[indx[1]])
  }
  # CLEAN UP
  rm(indx,possible_lib_paths)
}
# =====================================================================
library("optparse")
library("grf")

option_list = list(
  make_option(c("-p", "--prefix"), type="character", default=NULL, 
              help="File prefix", metavar="character"))

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

fprefix <- opt$prefix
summary_file <- paste(fprefix, "file_names.txt", sep="")
results_file <- paste(fprefix, "results.csv", sep="")
results_data <- read.csv(file=results_file, header=TRUE, sep=",")
con <- file(summary_file, open='r')
fnames <- readLines(con)
close(con)
n_files <- strtoi(fnames[1])

results1 <- results_data[, grepl("x[0-9]|TE_hat", colnames(results_data))]
results2 <- results_data[, grepl("x[0-9]|TE_hat", colnames(results_data))]
results3 <- results_data[, grepl("x[0-9]|TE_hat", colnames(results_data))]
results4 <- results_data[, grepl("x[0-9]|TE_hat", colnames(results_data))]

start_time <- Sys.time()
for (t in c(2:(n_files+1))){
  print(paste("Iteration", t-2))
  fname <- fnames[t]
  data <- read.csv(file=fname, header=TRUE, sep=",")
  test_data <- results_data[, grepl("W|x[0-9]", colnames(results_data))]
  te_col <- paste("TE_",t-2,sep="")
  
  ### Get treatment effects
  ### Comparison #1
  # Fit on W U x
  input <- data[, grepl("(W|x)[0-9]", colnames(data))]
  forest <- causal_forest(input, data$Y, data$T)
  tau_hat <- predict(forest, test_data)
  results1[te_col] = tau_hat$predictions
  
  ### Comparison #2
  # Fit on x
  input <- data[colnames(data)[grepl("x[0-9]", colnames(data))]]
  test_data <- results_data[colnames(results_data)[grepl("x[0-9]", colnames(results_data))]]
  forest <- causal_forest(input, data$Y, data$T)
  tau_hat <- predict(forest, test_data)
  results2[te_col] = tau_hat$predictions
  
  ### Comparison #3
  # Residualize on W, fit on x
  forest <- causal_forest(input, data$res_Y_W, data$res_T_W)
  tau_hat <- predict(forest, test_data)
  results3[te_col] <- tau_hat$predictions
  
  ### Comparison #4
  # Residualize on W U x, fit on x
  forest <- causal_forest(input, data$res_Y_Wx, data$res_T_Wx)
  tau_hat <- predict(forest, test_data)
  results4[te_col] <- tau_hat$predictions
  
  end_time <- Sys.time()
  print(end_time-start_time)
  start_time <- Sys.time()
}

f_out_name1 <- paste(fprefix, "GRF_Wx_results.csv", sep="")
f_out_name2 <- paste(fprefix, "GRF_x_results.csv", sep="")
f_out_name3 <- paste(fprefix, "GRF_res_W_results.csv", sep="")
f_out_name4 <- paste(fprefix, "GRF_res_Wx_results.csv", sep="")

write.csv(results1, file=f_out_name1, row.names=FALSE)
write.csv(results2, file=f_out_name2, row.names=FALSE)
write.csv(results3, file=f_out_name3, row.names=FALSE)
write.csv(results4, file=f_out_name4, row.names=FALSE)
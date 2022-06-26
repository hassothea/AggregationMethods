#### --------------------------------------------------------------------------- ####
#### -------- A kernel-based combined classification rule - Mojirsheibani (2000) ####
#### --------------------------------------------------------------------------- ####

#### Check if package "pacman" is already installed 
lookup_packages <- installed.packages()[,1]
if(!("fontawesome" %in% lookup_packages))
  install.packages("fontawesome")


#### To be installed or loaded
lookup_packages <- installed.packages()[,1]
if(!("pacman" %in% lookup_packages))
  install.packages("pacman")


# To be installed or loaded
pacman::p_load(magrittr)
pacman::p_load(tidyverse)

## package for "generateMachines"
pacman::p_load(tree)
pacman::p_load(nnet)
pacman::p_load(e1071)
pacman::p_load(randomForest)
pacman::p_load(FNN)
pacman::p_load(xgboost)
pacman::p_load(adabag)
pacman::p_load(keras)
pacman::p_load(pracma)
pacman::p_load(latex2exp)
pacman::p_load(plotly)
rm(lookup_packages)

# ------------------------------------------------------------------------------------------------ #

# Function: `setBasicParameter`
# -----------------------------

setBasicParameter <- function(k = 10,
                              ntree = 300,
                              mtry = NULL,
                              ker_svm = "radial",
                              deg_svm = 3,
                              mfinal_boost = 50,
                              boostrap = TRUE,
                              eta_xgb = 1, 
                              nrounds_xgb = 100, 
                              early_stop_xgb = NULL,
                              max_depth_xgb = 3,
                              param_xgb = NULL){
  return(list(
    k = k,
    ntree = ntree, 
    mtry = mtry, 
    ker_svm = ker_svm,
    deg_svm = deg_svm,
    mfinal_boost = mfinal_boost,
    boostrap = boostrap,
    eta_xgb = eta_xgb, 
    nrounds_xgb = nrounds_xgb, 
    early_stop_xgb = early_stop_xgb,
    max_depth_xgb = max_depth_xgb,
    param_xgb = param_xgb)
  )
}

# ------------------------------------------------------------------------------------------------ #

# Function: `generateMachines`
# ----------------------------
### Note: You may need to modify the function accordingly if you want to build different types of basic machines.

generateMachines <- function(train_input, 
                             train_response,
                             scale_input = FALSE,
                             machines = NULL,
                             splits = 0.5, 
                             basicMachineParam = setBasicParameter(),
                             silent = FALSE){
  k <- basicMachineParam$k 
  ntree <- basicMachineParam$ntree 
  mtry <- basicMachineParam$mtry
  ker_svm <- basicMachineParam$ker_svm
  deg_svm <- basicMachineParam$deg_svm
  mfinal_boost = basicMachineParam$mfinal_boost
  boostrap = basicMachineParam$boostrap
  eta_xgb <- basicMachineParam$eta_xgb 
  nrounds_xgb <- basicMachineParam$nrounds_xgb
  early_stop_xgb <- basicMachineParam$early_stop_xgb
  max_depth_xgb <- basicMachineParam$max_depth_xgb
  param_xgb <- basicMachineParam$param_xgb
  class_xgb <- unique(train_response)
  numberOfClasses <- length(class_xgb)
  if(is.null(param_xgb)){
    param_xgb <- list("objective" = "multi:softmax",
                      "eval_metric" = "mlogloss",
                      "num_class" = numberOfClasses+1)
  }
  
  # Packages
  pacman::p_load(nnet)
  pacman::p_load(e1071)
  pacman::p_load(tree)
  pacman::p_load(randomForest)
  pacman::p_load(FNN)
  pacman::p_load(xgboost)
  pacman::p_load(maboost)
  
  # Preparing data
  input_names <- colnames(train_input)
  input_size <- dim(train_input)
  df_input <- train_input_scale <- train_input
  if(scale_input){
    maxs <- map_dbl(.x = df_input, .f = max)
    mins <- map_dbl(.x = df_input, .f = min)
    train_input_scale <- scale(train_input, 
                               center = mins, 
                               scale = maxs - mins)
  }
  if(is.matrix(train_input_scale)){
    df_input <- as_tibble(train_input_scale)
    matrix_input <- train_input_scale
  } else{
    df_input <- train_input_scale
    matrix_input <- as.matrix(train_input_scale)
  }
  
  # Machines
  svm_machine <- function(x, pa = NULL){
    mod <- svm(x = df_train_x1, 
               y = train_y1,
               kernel = ker_svm,
               degree = deg_svm,
               type = "C-classification")
    res <- predict(mod, 
                   newdata = x)
    return(list(pred = res,
                model = mod))
  }
  tree_machine <- function(x, pa = NULL) {
    mod <- tree(as.formula(paste("train_y1~", 
                                 paste(input_names, 
                                       sep = "", 
                                       collapse = "+"), 
                                 collapse = "", 
                                 sep = "")), 
                data = df_train_x1)
    res <- predict(mod, x, type = 'class')
    return(list(pred = res,
                model = mod))
  }
  knn_machine <- function(x, k0) {
    mod <- knn(train = matrix_train_x1, 
               test = x, 
               cl = train_y1, 
               k = k0)
    return(list(pred = mod,
                model = k0))
  }
  RF_machine <- function(x, ntree0) {
    if(is.null(mtry)){
      mod <- randomForest(x = df_train_x1, 
                          y = train_y1, 
                          ntree = ntree0)
    }else{
      mod <- randomForest(x = df_train_x1, 
                          y = train_y1, 
                          ntree = ntree0, 
                          mtry = mtry)
    }
    res <- as.vector(predict(mod, x))
    return(list(pred = res,
                model = mod))
  }
  xgb_machine <- function(x, nrounds_xgb0){
    mod <- xgboost(data = matrix_train_x1,
                   label = train_y1,
                   params = param_xgb,
                   eta = eta_xgb,
                   early_stopping_rounds = early_stop_xgb,
                   max_depth = max_depth_xgb,
                   verbose = 0,
                   nrounds = nrounds_xgb0)
    res <- class_xgb[predict(mod, x)]
    return(list(pred = res,
                model = mod))
  }
  ada_machine <- function(x, mfinal0){
    data_tem <- cbind(df_train_x1, "target" = train_y1)
    mod_ <- boosting(target ~ ., 
                     data = data_tem,
                     mfinal = mfinal0,
                     boos = boostrap)
    res <- predict.boosting(mod_, 
                            newdata = as.data.frame(x))
    return(list(pred = res$class,
                model = mod_))
  }
  logit_machine <- function(x, pa = NULL){
    mod <- multinom(as.formula(paste("train_y1~", 
                                     paste(input_names, 
                                           sep = "", 
                                           collapse = "+"), 
                                     collapse = "", 
                                     sep = "")), 
                    data = df_train_x1,
                    trace = FALSE)
    res <- predict(mod, 
                   newdata = x)
    return(list(pred = res,
                model = mod))
  }
  # All machines
  all_machines <- list(knn = knn_machine, 
                       tree = tree_machine, 
                       rf = RF_machine,
                       logit = logit_machine,
                       svm = svm_machine,
                       adaboost = ada_machine,
                       xgb = xgb_machine)
  # All parameters
  all_parameters <- list(knn = k, 
                         tree = 1,
                         rf = ntree,
                         logit = NA,
                         svm = deg_svm,
                         adaboost = mfinal_boost,
                         xgb = nrounds_xgb)
  lookup_machines <- c("knn", "tree", "rf", "logit", "svm", "xgb", "adaboost")
  if(is.null(machines)){
    mach <- lookup_machines
  }else{
    mach <- map_chr(.x = machines,
                    .f = ~ match.arg(.x, lookup_machines))
  }
  # Extracting data
  M <- length(mach)
  size_D1 <- floor(splits*input_size[1])
  id_D1 <- logical(input_size[1])
  id_D1[sample(input_size[1], size_D1)] <- TRUE
  
  df_train_x1 <- df_input[id_D1,]
  matrix_train_x1 <- matrix_input[id_D1,]
  train_y1 <- train_response[id_D1]
  df_train_x2 <- df_input[!id_D1,]
  matrix_train_x2 <- matrix_input[!id_D1,]
  
  # Function to extract df and model from 'map' function
  extr_df <- function(x, nam, id){
    return(tibble("{nam}_{id}" := as.vector(pred_m[[x]]$pred)))
  }
  extr_mod <- function(x, id){
    return(pred_m[[x]]$model)
  }
  
  pred_D2 <- c()
  all_mod <- c()
  if(!silent){
    cat("\n* Building basic machines ...\n")
    cat("\t~ Progress:")
  }
  for(m in 1:M){
    if(mach[m] %in% c("knn", "xgb")){
      x0_test <-  matrix_train_x2
    } else {
      x0_test <- df_train_x2
    }
    if(is.null(all_parameters[[mach[m]]])){
      para_ <- 1
    }else{
      para_ <- all_parameters[[mach[m]]]
    }
    pred_m <-  map(para_,
                   .f = ~ all_machines[[mach[m]]](x0_test, .x))
    tem0 <- imap_dfc(.x = 1:length(para_), 
                     .f = ~ extr_df(x = .x, nam = mach[m], id = para_[.x]))
    tem1 <- map(.x = 1:length(para_), 
                .f = extr_mod)
    names(tem0) <- names(tem1) <- paste0(mach[m], 1:length(para_))
    pred_D2 <- bind_cols(pred_D2, as_tibble(tem0))
    all_mod[[mach[m]]] <- tem1
    if(!silent){
      cat(" ... ", round(m/M, 2)*100L,"%", sep = "")
    }
  }
  if(scale_input){
    return(list(predict2 = pred_D2,
                models = all_mod,
                id2 = !id_D1,
                train_data = list(train_input = train_input_scale, 
                                  train_response = train_response,
                                  classes = class_xgb),
                scale_max = maxs,
                scale_min = mins))
  } else{
    return(list(predict2 = pred_D2,
                models = all_mod,
                id2 = !id_D1,
                train_data = list(train_input = train_input_scale, 
                                  train_response = train_response,
                                  classes = class_xgb)))
  }
}

# ------------------------------------------------------------------------------------------------ #

# Function: `setGridParameter`
# ----------------------------

setGridParameter <- function(min_val = 1e-5, 
                             max_val = 0.5, 
                             n_val = 300, 
                             parameters = NULL,
                             print_result = TRUE,
                             figure = TRUE){
  return(list(min_val = min_val,
              max_val = max_val,
              n_val = n_val,
              parameters = parameters,
              print_result = print_result,
              figure = figure))
}

# ------------------------------------------------------------------------------------------------ #

# Function: `gridOptimizer`
# -------------------------

gridOptimizer <- function(obj_func,
                          setParameter = setGridParameter(),
                          naive = FALSE,
                          silent = FALSE,
                          ker = NULL){
  t0 <- Sys.time()
  if(!naive){
    if(is.null(setParameter$parameters)){
      param <- seq(setParameter$min_val,
                   setParameter$max_val,
                   length.out = setParameter$n_val)
    } else{
      param <- setParameter$parameters
    }
    risk <- param %>%
      map_dbl(.f = obj_func)
    id_opt <- which.min(risk)
    opt_ep <- param[id_opt]
    opt_risk <- risk[id_opt]
    if(setParameter$print_result & !silent){
      cat("\n* Grid search for",ker,"kernel...\n ~ observed parameter :", opt_ep)
    }
    if(setParameter$figure){
      tibble(x = param, 
             y = risk) %>%
        ggplot(aes(x = x, y = y)) +
        geom_line(color = "skyblue", size = 0.75) +
        geom_point(aes(x = opt_ep, y = opt_risk), color = "red") +
        geom_vline(xintercept = opt_ep, color = "red", linetype = "longdash") +
        labs(title = "Error as function of parameter", 
             x = "Parameter",
             y = "Error") -> p
      print(p)
    }
  } else{
    opt_ep = NA
    opt_risk = NA
    risk = NA
  }
  
  t1 <- Sys.time()
  return(list(
    opt_param = opt_ep,
    opt_error = opt_risk,
    all_risk = risk,
    run.time = difftime(t1, 
                        t0, 
                        units = "secs")[[1]])
  )
}

# ------------------------------------------------------------------------------------------------ #

# Function: `dist_matrix`
# -----------------------
dist_matrix <- function(basicMachines,
                        n_cv = 5,
                        id_shuffle = NULL){
  n <- nrow(basicMachines$predict2)
  n_each_fold <- floor(n/n_cv)
  # shuffled indices
  if(is.null(id_shuffle)){
    shuffle <- 1:(n_cv-1) %>%
      rep(n_each_fold) %>%
      c(., rep(n_cv, n - n_each_fold * (n_cv - 1))) %>%
      sample
  }else{
    shuffle <- id_shuffle
  }
  # the prediction matrix D_l
  df_ <- as.matrix(basicMachines$predict2)
  pair_dist <- function(M, N){
    res_ <- 1:nrow(N) %>%
      map_dfc(.f = (\(id) tibble('{{id}}' := rowSums(sweep(M, 2, N[id,], FUN = "!=")))))
    return(res_)
  }
  L <- 1:n_cv %>%
    map(.f = ~ pair_dist(df_[shuffle != .x,],
                         df_[shuffle == .x,]))
  return(list(dist = L, 
              id_shuffle = shuffle,
              n_cv = n_cv))
}

# ------------------------------------------------------------------------------------------------ #

# Fitting parameter
# -----------------
fit_parameter <- function(train_design, 
                          train_response,
                          scale_input = FALSE,
                          build_machine = TRUE,
                          machines = NULL, 
                          splits = 0.5, 
                          n_cv = 5,
                          inv_sigma = sqrt(.5),
                          alp = 2,
                          kernels = "gaussian",
                          setMachineParam = setBasicParameter(),
                          setGridParam = setGridParameter(),
                          silent = FALSE){
  kernels_lookup <- c("gaussian", "epanechnikov", "biweight", "triweight", "triangular", "naive")
  kernel_real <- kernels %>%
    map_chr(.f = ~ match.arg(.x, 
                             kernels_lookup))
  if(build_machine){
    mach2 <- generateMachines(train_input = train_design,
                              train_response = train_response,
                              scale_input = scale_input,
                              machines = machines,
                              splits = splits,
                              basicMachineParam = setMachineParam,
                              silent = silent)
  } else{
    mach2 <- list(predict2 = train_design,
                  models = colnames(train_design),
                  id2 = rep(TRUE, nrow(train_design)),
                  train_data = list(train_response = train_response,
                                    classes = unique(train_response)))                 
  }
  # distance matrix to compute loss function
  n_ker <- length(kernels)
  id_shuf <- NULL
  dist_all <- dist_matrix(basicMachines = mach2,
                          n_cv = n_cv)
  
  # Kernel functions
  # ================
  # Gaussian kernel
  gaussian_kernel <- function(.ep = .05,
                              .dist_matrix,
                              .train_response2,
                              .inv_sigma = sqrt(.5),
                              .alpha = 2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(exp(-(x*D)^(.alpha/2)*.inv_sigma^.alpha))
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, 
                                 id = .x, 
                                 D = .y))
    return(Reduce("+", temp)/.dist_matrix$n_cv)
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(.ep = .05,
                                  .dist_matrix,
                                  .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] = 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, 
                                 id = .x, 
                                 D = .y))
    return(Reduce("+", temp)/.dist_matrix$n_cv)
  }
  
  # Biweight
  biweight_kernel <- function(.ep = .05,
                              .dist_matrix,
                              .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] = 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, 
                                 id = .x, 
                                 D = .y))
    return(Reduce("+", temp)/.dist_matrix$n_cv)
  }
  
  # Triweight
  triweight_kernel <- function(.ep = .05,
                               .dist_matrix,
                               .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] = 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, 
                                 id = .x, 
                                 D = .y))
    return(Reduce("+", temp)/.dist_matrix$n_cv)
  }
  
  # Triangular
  triangular_kernel <- function(.ep = .05,
                                .dist_matrix,
                                .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] <- 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, 
                                 id = .x, 
                                 D = .y))
    return(Reduce("+", temp)/.dist_matrix$n_cv)
  }
  
  # error function
  error_cv <- function(x, 
                       .dist_matrix = NULL,
                       .kernel_func = NULL,
                       .train_response2 = NULL){
    res <- .kernel_func(.ep = x,
                        .dist_matrix = .dist_matrix,
                        .train_response2 = .train_response2)
    return(res/n_cv)
  }
  
  # list of kernel functions
  list_funs <- list(gaussian = gaussian_kernel,
                    epanechnikov = epanechnikov_kernel,
                    biweight = biweight_kernel,
                    triweight = triweight_kernel,
                    triangular = triangular_kernel,
                    naive = epanechnikov_kernel)
  
  # error for all kernel functions
  error_func <- kernel_real %>%
    map(.f = ~ (\(x) error_cv(x, 
                              .dist_matrix = dist_all,
                              .kernel_func = list_funs[[.x]],
                              .train_response2 = train_response[mach2$id2])))
  names(error_func) <- kernel_real
  
  # Optimization
  parameters <- map(.x = kernel_real,
                    .f = ~ gridOptimizer(obj_fun = error_func[[.x]],
                                         setParameter = setGridParam,
                                         naive = .x == "naive",
                                         silent = silent,
                                         ker = .x))
  names(parameters) <- kernel_real
  return(list(opt_parameters = parameters,
              add_parameters = list(scale_input = scale_input,
                                    inv_sigma = inv_sigma,
                                    alp = alp),
              basic_machines = mach2))
}

# ------------------------------------------------------------------------------------------------ #

# Prediction
# ----------

kernel_pred <- function(.h,
                        .y2, 
                        .distance, 
                        .kern = "gaussian",
                        .inv_sig = sqrt(.5), 
                        .alp = 2){
  dis <- as.matrix(.distance)
  # Kernel functions 
  # ================
  gaussian_kernel <- function(.ep,
                              .inv_sigma = .inv_sig,
                              .alpha = .alp){
    tem0 <- exp(- (.ep*dis)^(.alpha/2)*.inv_sig^.alpha)
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] = 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  # Biweight
  biweight_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] = 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  
  # Triweight
  triweight_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] = 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  
  # Triangular
  triangular_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] <- 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  # Naive
  naive_kernel <- function(.ep = NULL){
    y_hat <- map_dfc(.x = 1:ncol(dis),
                     .f = (\(x_) tibble("{x_}" := tapply(dis[, x_] == 0, 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  # Prediction
  kernel_list <- list(gaussian = gaussian_kernel,
                      epanechnikov = epanechnikov_kernel,
                      biweight = biweight_kernel,
                      triweight = triweight_kernel,
                      triangular = triangular_kernel,
                      naive = naive_kernel)
  res <- tibble(as.vector(kernel_list[[.kern]](.ep = .h)))
  names(res) <- .kern
  return(res)
}

# ------------------------------------------------------------------------------------------------ #

# Functions: `predict_agg`
# ------------------------

### Prediction
predict_agg <- function(fitted_models,
                        new_data,
                        test_response = NULL,
                        naive = FALSE){
  opt_param <- fitted_models$opt_parameters
  add_param <- fitted_models$add_parameters
  basic_mach <- fitted_models$basic_machines
  kern0 <- names(opt_param)
  new_data_ <- new_data
  # if basic machines are built
  if(is.list(basic_mach$models)){
    mat_input <- as.matrix(basic_mach$train_data$train_input)
    if(add_param$scale_input){
      new_data_ <- scale(new_data, 
                         center = basic_mach$scale_min, 
                         scale = basic_mach$scale_max - basic_mach$scale_min)
    }
    if(is.matrix(new_data_)){
      mat_test <- new_data_
      df_test <- as_tibble(new_data_)
    } else {
      mat_test <- as.matrix(new_data_)
      df_test <- new_data_
    }
    
    # Prediction test by basic machines
    built_models <- basic_mach$models
    pred_test <- function(meth){
      if(meth == "knn"){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := FNN::knn(train = mat_input[!basic_mach$id2,], 
                                                        test = mat_test, 
                                                        cl = basic_mach$train_data$train_response[!basic_mach$id2],
                                                        k = built_models[[meth]][[k]]))))
      }
      if(meth == "xgb"){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(basic_mach$train_data$classes[predict(built_models[[meth]][[k]], mat_test)]))))
      }
      if(meth == "adaboost"){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict.boosting(built_models[[meth]][[k]], as.data.frame(df_test))$class))))
      }
      if(!(meth %in% c("xgb", "knn", "adaboost"))){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict(built_models[[meth]][[k]], df_test, type = 'class')))))
      }
      colnames(pre) <- names(built_models[[meth]])
      return(pre)
    }
    pred_test_all <- names(built_models) %>%
      map_dfc(.f = pred_test)
  } else{
    pred_test_all <- new_data_
  }
  pred_test0 <- pred_test_all
  # Prediction train2
  pred_train_all <- basic_mach$predict2
  colnames(pred_test_all) <- colnames(pred_train_all)
  d_train <- dim(pred_train_all)
  d_test <- dim(pred_test_all)
  pred_test_mat <- as.matrix(pred_test_all)
  pred_train_mat <- as.matrix(pred_train_all)
  # Distance matrix
  dists <- 1:d_test[1] %>%
    map_dfc(.f = (\(id) tibble('{{id}}' := rowSums(sweep(pred_train_mat, 2, pred_test_mat[id,], FUN = "!=")))))
  prediction <- 1:length(kern0) %>% 
    map_dfc(.f = ~ kernel_pred(.h = opt_param[[kern0[.x]]]$opt_param,
                               .y2 = basic_mach$train_data$train_response[basic_mach$id2],
                               .distance = dists,
                               .kern = kern0[.x], 
                               .inv_sig = add_param$inv_sigma, 
                               .alp = add_param$alp))
  if(is.null(test_response)){
    return(list(fitted_aggregate = prediction,
                fitted_machine = pred_test_all))
  } else{
    error <- cbind(pred_test_all, prediction) %>%
      dplyr::mutate(y_test = test_response) %>%
      dplyr::summarise_all(.funs = ~ (. != y_test)) %>%
      dplyr::select(-y_test) %>%
      dplyr::summarise_all(.funs = ~ mean(.))
    return(list(fitted_aggregate = prediction,
                fitted_machine = pred_test0,
                mis_error = error,
                accuracy = 1 - error))
  }
}

# ------------------------------------------------------------------------------------------------ #

# Function: `kernelAggClass`
# -----------------------

kernelAggClass <- function(train_design, 
                           train_response,
                           test_design,
                           test_response = NULL,
                           scale_input = FALSE,
                           build_machine = TRUE,
                           machines = NULL, 
                           splits = 0.5, 
                           n_cv = 5,
                           inv_sigma = sqrt(.5),
                           alp = 2,
                           kernels = "gaussian",
                           setMachineParam = setBasicParameter(),
                           setGridParam = setGridParameter(),
                           silent = FALSE){
  # build machines + tune parameter
  fit_mod <- fit_parameter(train_design = train_design, 
                           train_response = train_response,
                           scale_input = scale_input,
                           build_machine = build_machine,
                           machines = machines, 
                           splits = splits, 
                           n_cv = n_cv,
                           inv_sigma = inv_sigma,
                           alp = alp,
                           kernels = kernels,
                           setMachineParam = setMachineParam,
                           setGridParam = setGridParam,
                           silent = silent)
  # prediction
  pred <- predict_agg(fitted_models = fit_mod,
                      new_data = test_design,
                      test_response = test_response)
  return(list(fitted_aggregate = pred$fitted_aggregate,
              fitted_machine = pred$fitted_machine,
              pred_train2 = fit_mod$basic_machines$predict2,
              opt_parameter = fit_mod$opt_parameters,
              mis_class = pred$mis_error,
              accuracy = pred$accuracy,
              kernels = kernels,
              ind_D2 = fit_mod$basic_machines$id2))
}

# ------------------------------------------------------------------------------------------------ #

#### ------------------------------------------------------------------------- ####
#### -------------------- MixCOBRA for classification ------------------------ ####
#### ------------------------------------------------------------------------- ####




#### Check if package "pacman" is already installed 

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



# Basic Machine generator
# -----------------------
## Function: `setBasicParameter_Mix`
## -----------------------------

setBasicParameter_Mix <- function(k = 10,
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

#### --------------------------------------------------------------------- ####

## Function: `generateMachines_Mix`
## ----------------------------

generateMachines_Mix <- function(train_input, 
                                 train_response,
                                 scale_input = FALSE,
                                 machines = NULL,
                                 splits = 0.5, 
                                 basicMachineParam = setBasicParameter_Mix(),
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
    return(list(fitted_remain = pred_D2,
                models = all_mod,
                id2 = !id_D1,
                train_data = list(train_input = train_input_scale, 
                                  train_response = train_response,
                                  classes = class_xgb),
                scale_max = maxs,
                scale_min = mins))
  } else{
    return(list(fitted_remain = pred_D2,
                models = all_mod,
                id2 = !id_D1,
                train_data = list(train_input = train_input_scale, 
                                  train_response = train_response,
                                  classes = class_xgb)))
  }
}

#### --------------------------------------------------------------------- ####


# Optimization algorithm
# ----------------------

### Function: `setGridParameter_Mix`
### ----------------------------

setGridParameter_Mix <- function(min_alpha = 1e-5,
                                 max_alpha = 5,
                                 min_beta = 0.1,
                                 max_beta = 50,
                                 n_alpha = 30,
                                 n_beta = 30,
                                 parameters = NULL,
                                 axes = c("alpha", "beta", "Risk"),
                                 title = NULL,
                                 print_result = TRUE,
                                 figure = TRUE){
  return(list(min_alpha = min_alpha,
              max_alpha = max_alpha,
              min_beta = min_beta,
              max_beta = max_beta,
              n_alpha = n_alpha,
              n_beta = n_beta,
              axes = axes,
              title = title,
              parameters = parameters,
              print_result = print_result,
              figure = figure))
}

#### --------------------------------------------------------------------- ####

### Function: `gridOptimizer_Mix`

gridOptimizer_Mix <- function(obj_func,
                              setParameter = setGridParameter_Mix(),
                              silent = FALSE){
  t0 <- Sys.time()
  if(is.null(setParameter$parameters)){
    param_list <- list(alpha =  rep(seq(setParameter$min_alpha, 
                                        setParameter$max_alpha,
                                        length.out = setParameter$n_alpha), 
                                    setParameter$n_beta),
                       beta =  rep(seq(setParameter$min_beta, 
                                       setParameter$max_beta,
                                       length.out = setParameter$n_beta),
                                   each = setParameter$n_alpha))
  } else{
    param_list <- list(alpha = rep(setParameter$parameters[[1]], 
                                   length(setParameter$parameters[[2]])),
                       beta = rep(setParameter$parameters[[2]], 
                                  each = length(setParameter$parameters[[1]])))
  }
  risk <- map2_dbl(.x = param_list$alpha,
                   .y = param_list$beta,
                   .f = ~ obj_func(c(.x, .y)))
  id_opt <- which.min(risk)
  opt_ep <- c(param_list$alpha[id_opt], param_list$beta[id_opt])
  opt_risk <- risk[id_opt]
  if(setParameter$print_result & !silent){
    cat("\n* Grid search algorithm...", "\n ~ Observed parameter: (alpha, beta) = (", opt_ep[1], 
        ", ", 
        opt_ep[2], ")", 
        sep = "")
  }
  if(setParameter$figure){
    if(is.null(setParameter$title)){
      tit <- paste("<b> Cross-validation risk as a function of</b> (",
                   setParameter$axes[1],",", 
                   setParameter$axes[2],
                   ")")
    } else{
      tit <- setParameter$title
    }
    fig <- tibble(alpha = param_list$alpha, 
                  beta = param_list$beta,
                  risk = risk) %>%
      plot_ly(x = ~alpha, y = ~beta, z = ~risk, type = "mesh3d") %>%
      add_trace(x = c(opt_ep[1], opt_ep[1]),
                y = c(0, opt_ep[2]),
                z = c(opt_risk, opt_risk),
                type = "scatter3d",
                mode = 'lines+markers',
                line = list( 
                  width = 2,
                  color = "#5E88FC", 
                  dash = TRUE),
                marker = list(
                  size = 4,
                  color = ~c("#5E88FC", "#38DE25")),
                name = paste("Optimal",setParameter$axes[1])) %>%
      add_trace(x = c(0, opt_ep[1]),
                y = c(opt_ep[2], opt_ep[2]),
                z = c(opt_risk, opt_risk),
                type = "scatter3d",
                mode = 'lines+markers',
                line = list( 
                  width = 2,
                  color = "#F31536", 
                  dash = TRUE),
                marker = list(
                  size = 4,
                  color = ~c("#F31536", "#38DE25")),
                name = paste("Optimal",setParameter$axes[2]))  %>%
      add_trace(x = opt_ep[1],
                y = opt_ep[2],
                z = opt_risk,
                type = "scatter3d",
                mode = 'markers',
                marker = list(
                  size = 5,
                  color = "#38DE25"),
                name = "Optimal point") %>%
      layout(title = list(text = tit,
                          x = 0.075, 
                          y = 0.925,
                          font = list(family = "Verdana",
                                      color = "#5E88FC")),
             legend = list(x = 100, y = 0.5),
             scene = list(xaxis = list(title = setParameter$axes[1]),
                          yaxis = list(title = setParameter$axes[2]),
                          zaxis = list( title = setParameter$axes[3])))
    print(fig)
  }
  t1 <- Sys.time()
  return(list(opt_param = opt_ep,
              opt_error = opt_risk,
              all_risk = risk,
              run.time = difftime(t1, t0, units = "secs")[[1]])
  )
}

#### --------------------------------------------------------------------- ####

## Function: `dist_matrix_Mix`
## -----------------------

dist_matrix_Mix <- function(basicMachines,
                            n_cv = 5,
                            kernel = "gausian",
                            id_shuffle = NULL,
                            output = TRUE){
  n <- nrow(basicMachines$fitted_remain)
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
  df_mach <- as.matrix(basicMachines$fitted_remain)
  df_input <- as.matrix(basicMachines$train_data$train_input[basicMachines$id2,])
  if(!(kernel %in% c("naive", "triangular"))){
    pair_dist_input <- function(M, N){
      n_N <- dim(N)
      n_M <- dim(M)
      res_ <- 1:nrow(N) %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums((M - matrix(rep(N[id,], n_M[1]), ncol = n_M[2], byrow = TRUE))^2)))))
      return(res_)
    }
  }
  if(kernel == "triangular"){
    pair_dist_input <- function(M, N){
      n_N <- dim(N)
      n_M <- dim(M)
      res_ <- 1:nrow(N) %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(abs(M - matrix(rep(N[id,], n_M[1]), ncol = n_M[2], byrow = TRUE)))))))
      return(res_)
    }
  }
  if(kernel == "naive"){
    pair_dist_input <- function(M, N){
      n_N <- dim(N)
      n_M <- dim(M)
      res_ <- 1:nrow(N) %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(apply(abs(M - matrix(rep(N[id,], n_M[1]), ncol = n_M[2], byrow = TRUE)), 1, max)))))
      return(res_)
    }
  }
  pair_dist_output <- function(M, N){
    res_ <- 1:nrow(N) %>%
      map_dfc(.f = (\(id) tibble('{{id}}' := rowSums(sweep(M, 2, N[id,], FUN = "!=")))))
    return(res_)
  }
  L1 <- 1:n_cv %>%
    map(.f = (\(x) pair_dist_input(df_input[shuffle != x,],
                                   df_input[shuffle == x,])))
  if(output){
    L2 <- 1:n_cv %>%
      map(.f = (\(x) pair_dist_output(df_mach[shuffle != x,],
                                      df_mach[shuffle == x,])))
  } else{
    L2 <- NA
  }
  return(list(dist_input = L1,
              dist_machine = L2,
              id_shuffle = shuffle,
              n_cv = n_cv))
}

#### --------------------------------------------------------------------- ####

## Fitting parameter
## -----------------

fit_parameter_Mix <- function(train_input, 
                              train_response,
                              train_predictions = NULL,
                              machines = NULL, 
                              scale_input = TRUE,
                              splits = 0.5, 
                              n_cv = 5,
                              inv_sigma = sqrt(.5),
                              alp = 2,
                              kernels = "gaussian",
                              setBasicMachineParam = setBasicParameter_Mix(),
                              setGridParam = setGridParameter_Mix(),
                              silent = FALSE){
  kernels_lookup <- c("gaussian", "epanechnikov", "biweight", "triweight", "triangular", "naive")
  kernel_real <- kernels %>%
    sapply(FUN = function(x) return(match.arg(x, kernels_lookup)))
  if(is.null(train_predictions)){
    mach2 <- generateMachines_Mix(train_input = train_input,
                                  train_response = train_response,
                                  scale_input = scale_input,
                                  machines = machines,
                                  splits = splits,
                                  basicMachineParam = setBasicMachineParam)
  }else{
    mach2 <- list(fitted_remain = train_predictions,
                  models = NULL,
                  id2 = rep(TRUE, nrow(train_input)),
                  train_data = list(train_input = train_input, 
                                    train_response = train_response,
                                    predict_remain_org = train_predictions,
                                    min_machine = NULL,
                                    max_machine = NULL,
                                    min_input = NULL,
                                    max_input = NULL))
    if(scale_input){
      min_ <- map_dbl(train_input, .f = min)
      max_ <- map_dbl(train_input, .f = max)
      mach2$train_data$min_input = min_
      mach2$train_data$max_input = max_
      mach2$train_data$train_input <- scale(train_input, 
                                            center = min_, 
                                            scale = max_ - min_)
    }
  }
  # distance matrix to compute loss function
  if_euclid <- FALSE
  id_euclid <- NULL
  n_ker <- length(kernels)
  dist_all <- list()
  id_shuf <- NULL
  out_ <- TRUE
  for (k_ in 1:n_ker){
    ker <- kernel_real[k_]
    if(ker == "naive"){
      dist_all[["naive"]] <- dist_matrix_Mix(basicMachines = mach2,
                                             n_cv = n_cv,
                                             kernel = "naive",
                                             id_shuffle = id_shuf,
                                             output = out_)
    } else{
      if(ker == "triangular"){
        dist_all[["triangular"]] <- dist_matrix_Mix(basicMachines = mach2,
                                                    n_cv = n_cv,
                                                    kernel = "triangular",
                                                    id_shuffle = id_shuf,
                                                    output = out_)
      } else{
        if(if_euclid){
          dist_all[[ker]] <- dist_all[[id_euclid]]
        } else{
          dist_all[[ker]] <- dist_matrix_Mix(basicMachines = mach2,
                                             n_cv = n_cv,
                                             kernel = ker,
                                             id_shuffle = id_shuf,
                                             output = out_)
          id_euclid <- ker
          if_euclid <- TRUE
        }
      }
    }
    id_shuf <- dist_all[[1]]$id_shuffle
    out_ <- FALSE
  }
  dist_output <- dist_all[[1]]$dist_machine
  # Kernel functions 
  # ================
  # Gaussian
  gaussian_kernel <- function(.ep,
                              .dist_matrix,
                              .train_response2,
                              .inv_sigma = inv_sigma,
                              .alpha = alp){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(exp(- (x[1]*D1+x[2]*D2)^(.alpha/2)*.inv_sigma^.alpha))
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = dist_output[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(.ep,
                                  .dist_matrix,
                                  .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] = 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = dist_output[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Biweight
  biweight_kernel <- function(.ep,
                              .dist_matrix,
                              .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] = 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = dist_output[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Triweight
  triweight_kernel <- function(.ep,
                               .dist_matrix,
                               .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] = 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = dist_output[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Triangular
  triangular_kernel <- function(.ep,
                                .dist_matrix,
                                .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] <- 0
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = dist_output[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Naive
  naive_kernel <- function(.ep,
                           .dist_matrix,
                           .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- (as.matrix((x[1]*D1+x[2]*D2)) < 1)
      y_hat <- map_dfc(.x = 1:ncol(tem0),
                       .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                           INDEX = .train_response2[.dist_matrix$id_shuffle != id],
                                                           FUN = sum)))) %>%
        map_chr(.f = (\(x) names(which.max(x))))
      return(mean(y_hat != .train_response2[.dist_matrix$id_shuffle == id]))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = dist_output[[.x]]))
    return(Reduce("+", temp))
  }
  
  # list of kernel functions
  list_funs <- list(gaussian = gaussian_kernel,
                    epanechnikov = epanechnikov_kernel,
                    biweight = biweight_kernel,
                    triweight = triweight_kernel,
                    triangular = triangular_kernel,
                    naive = naive_kernel)
  
  # error for all kernel functions
  error_func <- kernel_real %>%
    map(.f = ~ (\(x) list_funs[[.x]](.ep = x,
                                     .dist_matrix = dist_all[[.x]],
                                     .train_response2 = train_response[mach2$id2])/n_cv))
  names(error_func) <- kernel_real
  
  # Optimization
  parameters <- map(.x = kernel_real,
                    .f = ~ gridOptimizer_Mix(obj_fun = error_func[[.x]],
                                             setParameter = setGridParam,
                                             silent = silent))
  names(parameters) <- kernel_real
  return(list(opt_parameters = parameters,
              add_parameters = list(scale_input = scale_input,
                                    inv_sigma = inv_sigma,
                                    alp = alp),
              basic_machines = mach2))
}


#### --------------------------------------------------------------------- ####

# Prediction 
# ----------

## Kernel functions
## ----------------

kernel_pred_Mix <- function(theta,
                            .y2, 
                            .dist1,
                            .dist2,
                            .kern = "gaussian",
                            .inv_sig = sqrt(.5), 
                            .alp = 2){
  distD <- as.matrix(theta[1]*.dist1+theta[2]*.dist1)
  # Kernel functions
  # ================
  gaussian_kernel <- function(D,
                              .inv_sigma = .inv_sig,
                              .alpha = .alp){
    tem0 <- exp(- D^(.alpha/2)*.inv_sig^.alpha)
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] = 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  # Biweight
  biweight_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] = 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  
  # Triweight
  triweight_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] = 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  
  # Triangular
  triangular_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] <- 0
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
                                                         INDEX = .y2,
                                                         FUN = sum)))) %>%
      map_chr(.f = (\(x) names(which.max(x))))
    return(as.vector(y_hat))
  }
  # Naive
  naive_kernel <- function(D){
    tem0 <- (D < 1)
    y_hat <- map_dfc(.x = 1:ncol(tem0),
                     .f = (\(x_) tibble("{x_}" := tapply(tem0[, x_], 
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
  res <- tibble(as.vector(kernel_list[[.kern]](D = distD)))
  names(res) <- .kern
  return(res)
}

#### --------------------------------------------------------------------- ####

## Functions: `predict_Mix`
## ------------------------

predict_Mix <- function(fitted_models,
                        new_data,
                        new_pred = NULL,
                        test_response = NULL){
  opt_param <- fitted_models$opt_parameters
  add_param <- fitted_models$add_parameters
  basic_mach <- fitted_models$basic_machines
  kern0 <- names(opt_param)
  new_data_ <- as.matrix(new_data)
  mat_input <- as.matrix(basic_mach$train_data$train_input)
  # if basic machines are built
  if(is.list(basic_mach$models)){
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
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(basic_mach$train_data$classes[predict(built_models[[meth]][[k]],
                                                                                               mat_test)]))))
      }
      if(meth == "adaboost"){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict.boosting(built_models[[meth]][[k]], 
                                                                          as.data.frame(df_test))$class))))
      }
      if(!(meth %in% c("xgb", "knn", "adaboost"))){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict(built_models[[meth]][[k]], 
                                                                 df_test, type = 'class')))))
      }
      colnames(pre) <- names(built_models[[meth]])
      return(pre)
    }
    pred_test_all <- names(built_models) %>%
      map_dfc(.f = pred_test)
  } else{
    pred_test_all <- new_pred
  }
  # Prediction train2
  pred_train_all <- basic_mach$fitted_remain
  colnames(pred_test_all) <- colnames(pred_train_all)
  d_train <- dim(pred_train_all)
  d_test <- dim(pred_test_all)
  d_train_input <- dim(mat_input[basic_mach$id2,])
  d_test_input <- dim(new_data_)
  pred_test_mat <- as.matrix(pred_test_all)
  pred_train_mat <- as.matrix(pred_train_all)
  # Distance matrix
  dist_mat <- function(kernel = "gausian"){
    res_1 <- res_2 <- NULL
    if(!(kernel %in% c("naive", "triangular"))){
      res_1 <- 1:d_test_input[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(sweep(mat_input[basic_mach$id2,], MARGIN = 2, new_data_[id,]))^2))))
    }
    if(kernel == "triangular"){
      res_1 <- 1:d_test_input[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(abs(sweep(mat_input[basic_mach$id2,], MARGIN = 2, new_data_[id,])))))))
    }
    if(kernel == "naive"){
      res_1 <- 1:d_test_input[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(apply(abs(sweep(mat_input[basic_mach$id2,], MARGIN = 2, new_data_[id,])), 1, max)))))
    }
    return(dist_input = res_1)
  }
  dist_input <- 1:length(kern0) %>%
    map(.f = ~ dist_mat(kern0[.x]))
  dist_preds <- 1:d_test[1] %>%
    map_dfc(.f = (\(id) tibble('{{id}}' := rowSums(sweep(pred_train_mat, 2, 
                                                         pred_test_mat[id,], 
                                                         FUN = "!=")))))
  prediction <- 1:length(kern0) %>% 
    map_dfc(.f = ~ kernel_pred_Mix(theta = opt_param[[kern0[.x]]]$opt_param,
                                   .y2 = basic_mach$train_data$train_response[basic_mach$id2],
                                   .dist1 = dist_input[[.x]],
                                   .dist2 = dist_preds,
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
                fitted_machine = pred_test_all,
                mis_error = error,
                accuracy = 1 - error))
  }
}

#### --------------------------------------------------------------------- ####

# Function : `MixCobraReg` 
# ========================

MixCobraClass <- function(train_input, 
                          train_response,
                          test_input,
                          train_predictions = NULL,
                          test_predictions = NULL,
                          test_response = NULL,
                          machines = NULL, 
                          scale_input = TRUE,
                          splits = 0.5, 
                          n_cv = 5,
                          inv_sigma = sqrt(.5),
                          alp = 2,
                          kernels = "gaussian",
                          setBasicMachineParam = setBasicParameter_Mix(),
                          setGridParam = setGridParameter_Mix(),
                          silent = FALSE){
  # build machines + tune parameter
  fit_mod <- fit_parameter_Mix(train_input = train_input, 
                               train_response = train_response,
                               train_predictions = train_predictions,
                               machines = machines, 
                               scale_input = scale_input,
                               splits = splits, 
                               n_cv = n_cv,
                               inv_sigma = inv_sigma,
                               alp = alp,
                               kernels = kernels,
                               setBasicMachineParam = setBasicMachineParam,
                               setGridParam = setGridParam,
                               silent = silent)
  # prediction
  pred <- predict_Mix(fitted_models = fit_mod,
                      new_data = test_input,
                      new_pred = test_predictions,
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

#### --------------------------------------------------------------------- ####

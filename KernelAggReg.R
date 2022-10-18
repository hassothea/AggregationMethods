#### ------------------------------------------------------------------- ####
#### -------- A kernel-based consensual aggregation method - Has. (2021) ####
#### ------------------------------------------------------------------- ####

#### Check if package "pacman" is already installed 
lookup_packages <- installed.packages()[,1]
if(!("pacman" %in% lookup_packages))
  install.packages("pacman")


#### To be installed or loaded
pacman::p_load(magrittr)
pacman::p_load(tidyverse)

#### package for "generateMachines"
pacman::p_load(tree)
pacman::p_load(glmnet)
pacman::p_load(randomForest)
pacman::p_load(FNN)
pacman::p_load(xgboost)
pacman::p_load(keras)
pacman::p_load(pracma)
pacman::p_load(latex2exp)
pacman::p_load(plotly)
rm(lookup_packages)

# ------------------------------------------------------------------------------------------------ #

# Function: `setBasicParameter`
# -----------------------------

setBasicParameter <- function(lambda = NULL,
                              k = 10, 
                              ntree = 300, 
                              mtry = NULL, 
                              eta_xgb = 1, 
                              nrounds_xgb = 100, 
                              early_stop_xgb = NULL,
                              max_depth_xgb = 3){
  return(list(
    lambda = lambda,
    k = k,
    ntree = ntree, 
    mtry = mtry, 
    eta_xgb = eta_xgb, 
    nrounds_xgb = nrounds_xgb, 
    early_stop_xgb = early_stop_xgb,
    max_depth_xgb = max_depth_xgb)
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
                             basicMachineParam = setBasicParameter()){
  lambda = basicMachineParam$lambda
  k <- basicMachineParam$k 
  ntree <- basicMachineParam$ntree 
  mtry <- basicMachineParam$mtry
  eta_xgb <- basicMachineParam$eta_xgb 
  nrounds_xgb <- basicMachineParam$nrounds_xgb
  early_stop_xgb <- basicMachineParam$early_stop_xgb
  max_depth_xgb <- basicMachineParam$max_depth_xgb
  
  # Packages
  pacman::p_load(tree)
  pacman::p_load(glmnet)
  pacman::p_load(randomForest)
  pacman::p_load(FNN)
  pacman::p_load(xgboost)
  # pacman::p_load(keras)
  
  # Preparing data
  input_names <- colnames(train_input)
  input_size <- dim(train_input)
  df_input <- train_input_scale <- train_input
  if(scale_input){
    maxs <- map_dbl(.x = df_input, .f = max)
    mins <- map_dbl(.x = df_input, .f = min)
    train_input_scale <- scale(train_input, center = mins, scale = maxs - mins)
  }
  if(is.matrix(train_input_scale)){
    df_input <- as_tibble(train_input_scale)
    matrix_input <- train_input_scale
  } else{
    df_input <- train_input_scale
    matrix_input <- as.matrix(train_input_scale)
  }
  
  # Machines
  lasso_machine <- function(x, lambda0){
    if(is.null(lambda)){
      cv <- cv.glmnet(matrix_train_x1, train_y1, alpha = 1, lambda = 10^(seq(-3,2,length.out = 50)))
      mod <- glmnet(matrix_train_x1, train_y1, alpha = 1, lambda = cv$lambda.min)
    } else{
      mod <- glmnet(matrix_train_x1, train_y1, alpha = 1, lambda = lambda0)
    }
    res <- predict.glmnet(mod, newx = x)
    return(list(pred = res,
                model = mod))
  }
  ridge_machine <- function(x, lambda0){
    if(is.null(lambda)){
      cv <- cv.glmnet(matrix_train_x1, train_y1, alpha = 0, lambda = 10^(seq(-3,2,length.out = 50)))
      mod <- glmnet(matrix_train_x1, train_y1, alpha = 0, lambda = cv$lambda.min)
    } else{
      mod <- glmnet(matrix_train_x1, train_y1, alpha = 0, lambda = lambda0)
    }
    res <- predict.glmnet(mod, newx = x)
    return(list(pred = res,
                model = mod))
  }
  tree_machine <- function(x, pa = NULL) {
    mod <- tree(train_y1 ~., 
                data = df_train_x1)
    res <- as.vector(predict(mod, x))
    return(list(pred = res,
                model = mod))
  }
  knn_machine <- function(x, k0) {
    mod <- knn.reg(train = matrix_train_x1, test = x, y = train_y1, k = k0)
    res = mod$pred
    return(list(pred = res,
                model = k0))
  }
  RF_machine <- function(x, ntree0) {
    if(is.null(mtry)){
      mod <- randomForest(x = df_train_x1, y = train_y1, ntree = ntree0)
    }else{
      mod <- randomForest(x = df_train_x1, y = train_y1, ntree = ntree0, mtry = mtry)
    }
    res <- as.vector(predict(mod, x))
    return(list(pred = res,
                model = mod))
  }
  xgb_machine = function(x, nrounds_xgb0){
    mod <- xgboost(data = matrix_train_x1, 
                   label = train_y1, 
                   eta = eta_xgb,
                   nrounds = nrounds_xgb0,
                   objective = "reg:squarederror",
                   early_stopping_rounds = early_stop_xgb,
                   max_depth = max_depth_xgb,
                   verbose = 0)
    res <- predict(mod, x)
    return(list(pred = res,
                model = mod))
  }
  
  # All machines
  all_machines <- list(lasso = lasso_machine, 
                       ridge = ridge_machine, 
                       knn = knn_machine, 
                       tree = tree_machine, 
                       rf = RF_machine,
                       xgb = xgb_machine)
  # All parameters
  all_parameters <- list(lasso = lambda, 
                         ridge = lambda, 
                         knn = k, 
                         tree = 1, 
                         rf = ntree,
                         xgb = nrounds_xgb)
  if(is.null(machines)){
    mach <- c("lasso", "ridge", "knn", "tree", "rf", "xgb")
  }else{
    mach <- machines
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
  extr_df <- function(x, id){
    return(tibble("r_{{id}}":= as.vector(pred_m[[x]]$pred)))
  }
  extr_mod <- function(x, id){
    return(pred_m[[x]]$model)
  }
  
  pred_D2 <- c()
  all_mod <- c()
  cat("\n* Building basic machines ...\n")
  cat("\t~ Progress:")
  for(m in 1:M){
    if(mach[m] %in% c("tree", "rf")){
      x0_test <- df_train_x2
    } else {
      x0_test <- matrix_train_x2
    }
    if(is.null(all_parameters[[mach[m]]])){
      para_ <- 1
    }else{
      para_ <- all_parameters[[mach[m]]]
    }
    pred_m <-  map(para_, 
                   .f = ~ all_machines[[mach[m]]](x0_test, .x))
    tem0 <- imap_dfc(.x = 1:length(para_), 
                     .f = extr_df)
    tem1 <- imap(.x = 1:length(para_), 
                 .f = extr_mod)
    names(tem0) <- names(tem1) <- paste0(mach[m], 1:length(para_))
    pred_D2 <- bind_cols(pred_D2, as_tibble(tem0))
    all_mod[[mach[m]]] <- tem1
    cat(" ... ", round(m/M, 2)*100L,"%", sep = "")
  }
  if(scale_input){
    return(list(predict2 = pred_D2,
                models = all_mod,
                id2 = !id_D1,
                train_data = list(train_input = train_input_scale, 
                                  train_response = train_response),
                scale_max = maxs,
                scale_min = mins))
  } else{
    return(list(predict2 = pred_D2,
                models = all_mod,
                id2 = !id_D1,
                train_data = list(train_input = train_input_scale, 
                                  train_response = train_response)))
  }
}

# ------------------------------------------------------------------------------------------------ #

# Function: `setGradParameter`
# ---------------------------

setGradParameter <- function(val_init = NULL, 
                             n_tries = 10, 
                             rate = 'linear', 
                             min_val = 1e-3,
                             max_val = 5,
                             max_iter = 300, 
                             print_step = TRUE, 
                             print_result = TRUE,
                             figure = TRUE, 
                             coef_auto = 0.0001,
                             coef_log = 1,
                             coef_sqrt = 1,
                             coef_lm = 1,
                             deg_poly = 2,
                             base_exp = 1.5,
                             threshold = 1e-10) {
  return(
    list(val_init = val_init,
         n_tries = n_tries,
         rate = rate,
         min_val = min_val,
         max_val = max_val,
         max_iter = max_iter,
         print_step = print_step,
         print_result = print_result,
         figure = figure,
         coef_auto = coef_auto,
         coef_log = coef_log,
         coef_sqrt = coef_sqrt,
         coef_lm = coef_lm,
         deg_poly = deg_poly,
         base_exp = base_exp,
         threshold = threshold
    )
  )
}

# ------------------------------------------------------------------------------------------------ #

# Function: `gradOptimizer`
# -------------------------
gradOptimizer <- function(obj_fun,
                          setParameter = setGradParameter()) {
  start.time <- Sys.time()
  # Optimization step:
  # ==================
  spec_print <- function(x) return(ifelse(x > 1e-6, 
                                          format(x, digit = 6, nsmall = 6), 
                                          format(x, scientific = TRUE, digit = 6, nsmall = 6)))
  collect_val <- c()
  gradients <- c()
  if (is.null(setParameter$val_init)){
    val_params <- seq(setParameter$min_val, 
                      setParameter$max_val, 
                      length.out = setParameter$n_tries)
    tem <- map_dbl(.x = val_params, .f = obj_fun)
    val <- val0 <- val_params[which.min(tem)]
    grad_ <- pracma::grad(
      f = obj_fun, 
      x0 = val0, 
      heps = .Machine$double.eps ^ (1 / 3))
  } else{
    val <- val0 <- setParameter$val_init
    grad_ <- pracma::grad(
      f = obj_fun, 
      x0 = val0, 
      heps = .Machine$double.eps ^ (1 / 3))
  }
  if(setParameter$print_step){
    cat("\n* Gradient descent algorithm ...")
    cat("\n  Step\t|  Parameter\t|  Gradient\t|  Threshold \n")
    cat(" ", rep("-", 51), sep = "")
    cat("\n   0 \t| ", spec_print(val0),
        "\t| ", spec_print(grad_), 
        " \t| ", setParameter$threshold, "\n")
    cat(" ", rep("-",51), sep = "")
  }
  if (is.numeric(setParameter$rate)){
    lambda0 <- setParameter$rate / abs(grad_)
    rate_GD <- "auto"
  } else{
    r0 <- setParameter$coef_auto / abs(grad_)
    # Rate functions
    rate_func <- list(auto = r0, 
                      logarithm = function(i)  setParameter$coef_log * log(2 + i) * r0,
                      sqrtroot = function(i) setParameter$coef_sqrt * sqrt(i) * r0,
                      linear = function(i) setParameter$coef_lm * (i) * r0,
                      polynomial = function(i) i ^ setParameter$deg_poly * r0,
                      exponential = function(i) setParameter$base_exp ^ i * r0)
    rate_GD <- match.arg(setParameter$rate, 
                         c("auto", 
                           "logarithm", 
                           "sqrtroot", 
                           "linear", 
                           "polynomial", 
                           "exponential"))
    lambda0 <- rate_func[[rate_GD]]
  }
  i <- 0
  if (is.numeric(setParameter$rate) | rate_GD == "auto") {
    while (i < setParameter$max_iter) {
      if(is.na(grad_)){
        val0 <- runif(1, val/2, 3*val/2)
        grad_ = pracma::grad(
          f = obj_fun, 
          x0 = val0, 
          heps = .Machine$double.eps ^ (1 / 3)
        )
      }
      val <- val0 - lambda0 * grad_
      if(val < 0){
        val <- val0/2
        lambda0 <- lambda0/2
      }
      if(i > 5){
        if(sign(grad_) * sign(grad0) < 0){
          lambda0 = lambda0 / 2
        }
      }
      relative <- abs((val - val0) / val0)
      test_threshold <- max(relative, abs(grad_))
      if (test_threshold > setParameter$threshold){
        val0 <- val
        grad0 <- grad_
      } else{
        break
      }
      grad_ = pracma::grad(
        f = obj_fun, 
        x0 = val0, 
        heps = .Machine$double.eps ^ (1 / 3)
      )
      i <- i + 1
      if(setParameter$print_step){
        cat("\n  ", i, "\t| ", spec_print(val), 
            "\t| ", spec_print(grad_), 
            "\t| ", test_threshold, "\r")
      }
      collect_val <- c(collect_val, val)
      gradients <- c(gradients, grad_)
    }
  }
  else{
    while (i < setParameter$max_iter) {
      if(is.na(grad_)){
        val0 <- runif(1, val/2, 3*val/2)
        grad_ = pracma::grad(
          f = obj_fun, 
          x0 = val0, 
          heps = .Machine$double.eps ^ (1 / 3)
        )
      }
      val <- val0 - lambda0(i) * grad_
      if(val < 0){
        val <- val0 / 2
        r0 <- r0 / 2
      }
      if(i > 5){
        if(sign(grad_)*sign(grad0) < 0)
          r0 <- r0 / 2
      }
      relative <- abs((val - val0) / val0)
      test_threshold <- max(relative, abs(grad_))
      if (test_threshold > setParameter$threshold){
        val0 <- val
        grad0 <- grad_
      }
      else{
        break
      }
      grad_ <- pracma::grad(
        f = obj_fun, 
        x0 = val0, 
        heps = .Machine$double.eps ^ (1 / 3)
      )
      i <- i + 1
      if(setParameter$print_step){
        cat("\n  ", i, "\t| ", spec_print(val), 
            "\t| ", spec_print(grad_), 
            "\t| ", test_threshold, "\r")
      }
      collect_val <- c(collect_val, val)
      gradients <- c(gradients, grad_)
    }
  }
  opt_ep <- val
  opt_risk <- obj_fun(opt_ep)
  if(setParameter$print_step){
    cat(rep("-", 55), sep = "")
    if(grad_ == 0){
      cat("\n Stopped| ", spec_print(val), 
          "\t| ", 0, 
          "\t\t| ", test_threshold)
    }else{
      cat("\n Stopped| ", spec_print(val), 
          "\t| ", spec_print(grad_), 
          "\t| ", test_threshold)
    } 
  }
  if(setParameter$print_result){
    cat("\n ~ Observed parameter:", opt_ep, " in",i, "iterations.")
  }
  if (setParameter$figure) {
    siz <- length(collect_val)
    tibble(x = 1:siz,
           y = collect_val,
           gradient = gradients) %>%
      ggplot(aes(x = x, y = y)) +
      geom_line(mapping = aes(color = gradient), size = 1) +
      geom_point(aes(x = length(x), y = opt_ep), color = "red") +
      geom_hline(yintercept = opt_ep, color = "red", linetype = "longdash") +
      labs(title = "Gradient steps",
           x = "Iteration",
           y = "Parameter")-> p
    print(p)
  }
  end.time = Sys.time()
  return(list(
    opt_param = opt_ep,
    opt_error = opt_risk,
    all_grad = gradients,
    all_param = collect_val,
    run_time = difftime(end.time, 
                        start.time, 
                        units = "secs")[[1]]
  ))
}

# ------------------------------------------------------------------------------------------------ #

# Function: `setGridParameter`
# ----------------------------

setGridParameter <- function(min_val = 1e-5, 
                             max_val = 5, 
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
                          setParameter = setGridParameter()){
  t0 <- Sys.time()
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
  if(setParameter$print_result){
    cat("\n* Grid search algorithm...", "\n ~ Observed parameter :", opt_ep)
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
           y = "Error")-> p
    print(p)
  }
  t1 <- Sys.time()
  return(list(
    opt_param = opt_ep,
    opt_error = opt_risk,
    all_risk = risk,
    run_time = difftime(t1, 
                        t0, 
                        units = "secs")[[1]])
  )
}

# ------------------------------------------------------------------------------------------------ #

# Function: `dist_matrix`
# -----------------------
dist_matrix <- function(basicMachines,
                        n_cv = 5,
                        kernel = "gaussian",
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
  if(! (kernel %in% c("naive", "triangular"))){
    pair_dist <- function(M, N){
      n_N <- dim(N)
      n_M <- dim(M)
      res_ <- 1:nrow(N) %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums((M - matrix(rep(N[id,], n_M[1]), ncol = n_M[2], byrow = TRUE))^2)))))
      return(res_)
    }
  }
  if(kernel == "triangular"){
    pair_dist <- function(M, N){
      n_N <- dim(N)
      n_M <- dim(M)
      res_ <- 1:nrow(N) %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(abs(M - matrix(rep(N[id,], n_M[1]), ncol = n_M[2], byrow = TRUE)))))))
      return(res_)
    }
  }
  if(kernel == "naive"){
    pair_dist <- function(M, N){
      n_N <- dim(N)
      n_M <- dim(M)
      res_ <- 1:nrow(N) %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(apply(abs(M - matrix(rep(N[id,], n_M[1]), ncol = n_M[2], byrow = TRUE)), 1, max)))))
      return(res_)
    }
  }
  L <- 1:n_cv %>%
    map(.f = (\(x) pair_dist(df_[shuffle != x,],
                             df_[shuffle == x,])))
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
                          scale_machine = FALSE,
                          build_machine = TRUE,
                          machines = NULL, 
                          splits = 0.5, 
                          n_cv = 5,
                          sig = 3,
                          alp = 2,
                          kernels = "gaussian",
                          optimizeMethod = "grad",
                          setBasicMachineParam = setBasicParameter(),
                          setGradParam = setGradParameter(),
                          setGridParam = setGridParameter()){
  kernels_lookup <- c("gaussian", "epanechnikov", "biweight", "triweight", "triangular", "naive", "c.expo", "expo")
  kernel_real <- kernels %>%
    sapply(FUN = function(x) return(match.arg(x, kernels_lookup)))
  if(build_machine){
    mach2 <- generateMachines(train_input = train_design,
                              train_response = train_response,
                              scale_input = scale_input,
                              machines = machines,
                              splits = splits,
                              basicMachineParam = setBasicMachineParam)
  } else{
    mach2 <- list(predict2 = train_design,
                  models = colnames(train_design),
                  id2 = rep(TRUE, nrow(train_design)),
                  train_data = list(train_response = train_response))                  
  }
  maxs_ <- mins_ <- NULL
  if(scale_machine){
    maxs_ <- map_dbl(.x = mach2$predict2, .f = max)
    mins_ <- map_dbl(.x = mach2$predict2, .f = min)
    mach2$predict2 <- scale(mach2$predict2, 
                            center = mins_, 
                            scale = maxs_ - mins_)
  }
  # distance matrix to compute loss function
  if_euclid <- FALSE
  id_euclid <- NULL
  n_ker <- length(kernels)
  dist_all <- list()
  id_shuf <- NULL
  for (k_ in 1:n_ker) {
    ker <- kernel_real[k_]
    if(ker == "naive"){
      dist_all[["naive"]] <- dist_matrix(basicMachines = mach2,
                                         n_cv = n_cv,
                                         kernel = "naive",
                                         id_shuffle = id_shuf)
    } else{
      if(ker == "triangular"){
        dist_all[["triangular"]] <- dist_matrix(basicMachines = mach2,
                                                n_cv = n_cv,
                                                kernel = "triangular",
                                                id_shuffle = id_shuf)
      } else{
        if(if_euclid){
          dist_all[[ker]] <- dist_all[[id_euclid]]
        } else{
          dist_all[[ker]] <- dist_matrix(basicMachines = mach2,
                                         n_cv = n_cv,
                                         kernel = ker,
                                         id_shuffle = id_shuf)
          id_euclid <- ker
          if_euclid <- TRUE
        }
      }
    }
    id_shuf <- dist_all[[1]]$id_shuffle
  }
  
  # Kernel functions 
  # ================
  # expo
  expo_kernel <- function(.ep = .05,
                          .dist_matrix,
                          .train_response2,
                          .alpha = alp){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(exp(- (x*D)^.alpha))
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv,
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  # C_expo
  c.expo_kernel <- function(.ep = .05,
                            .dist_matrix,
                            .train_response2,
                            .sigma = sig,
                            .alpha = alp){
    kern_fun <- function(x, id, D){
      tem0 <- x*D
      tem0[tem0 < .sigma] <- 0
      tem1 <- as.matrix(exp(- tem0^.alpha))
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem1/colSums(tem1)
      y_hat[is.na(y_hat)] <- 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv,
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  
  # Gaussian
  gaussian_kernel <- function(.ep = .05,
                              .dist_matrix,
                              .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(exp(- (x*D)/2))
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv,
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(.ep = .05,
                                  .dist_matrix,
                                  .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] = 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  
  # Biweight
  biweight_kernel <- function(.ep = .05,
                              .dist_matrix,
                              .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] = 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0^2/colSums(tem0^2)
      y_hat[is.na(y_hat)] <- 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  
  # Triweight
  triweight_kernel <- function(.ep = .05,
                               .dist_matrix,
                               .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] = 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0^3/colSums(tem0^3)
      y_hat[is.na(y_hat)] <- 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  
  # Triangular
  triangular_kernel <- function(.ep = .05,
                                .dist_matrix,
                                .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- as.matrix(1- x*D)
      tem0[tem0 < 0] <- 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      y_hat[is.na(y_hat)] = 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
  }
  
  # Naive
  naive_kernel <- function(.ep = .05,
                           .dist_matrix,
                           .train_response2){
    kern_fun <- function(x, id, D){
      tem0 <- (as.matrix(x*D) < 1)
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      y_hat[is.na(y_hat)] = 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map2(.x = 1:.dist_matrix$n_cv, 
                 .y = .dist_matrix$dist, 
                 .f = ~ kern_fun(x = .ep, id = .x, D = .y))
    return(Reduce("+", temp))
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
                    naive = naive_kernel,
                    expo = expo_kernel,
                    c.expo = c.expo_kernel)
  
  # error for all kernel functions
  error_func <- kernel_real %>%
    map(.f = ~ (\(t) error_cv(t, 
                              .dist_matrix = dist_all[[.x]],
                              .kernel_func = list_funs[[.x]],
                              .train_response2 = train_response[mach2$id2])))
  names(error_func) <- kernels
  # list of prameter setup
  list_param <- list(grad = setGradParam,
                     GD = setGradParam,
                     grid = setGridParam)
  # list of optimizer
  list_optimizer <- list(grad = gradOptimizer,
                         GD = gradOptimizer,
                         grid = gridOptimizer)
  optMethods <- optimizeMethod
  if(length(kernels) != length(optMethods)){
    warning("* kernels and optimization methods differ in sides! Grid search will be used!")
    optMethods = rep("grid", length(kernels))
  }
  
  # Optimization
  parameters <- map2(.x = kernels,
                     .y = optMethods,
                     .f = ~ list_optimizer[[.y]](obj_fun = error_func[[.x]],
                                                 setParameter = list_param[[.y]]))
  names(parameters) <- paste0(kernel_real, "_", optMethods)
  return(list(opt_parameters = parameters,
              add_parameters = list(scale_input = scale_input,
                                    scale_machine = scale_machine,
                                    sig = sig,
                                    alp = alp,
                                    opt_methods = optimizeMethod),
              basic_machines = mach2,
              .scale = list(min = mins_,
                            max = maxs_)))
}

# ------------------------------------------------------------------------------------------------ #

# Prediction
# ----------

kernel_pred <- function(epsilon,
                        .y2, 
                        .distance, 
                        .kern = "gaussian",
                        .sig = 3, 
                        .alp = 2,
                        .meth = NA){
  dis <- as.matrix(.distance)
  # Kernel functions 
  # ================
  expo_kernel <- function(.ep,
                          .alpha = .alp){
    tem0 <- exp(- (.ep*dis)^.alpha)
    y_hat <- .y2 %*% tem0/colSums(tem0)
    return(t(y_hat))
  }
  c.expo_kernel <- function(.ep,
                            .sigma = .sig,
                            .alpha = .alp){
    tem0 <- .ep*dis
    tem0[tem0 < .sigma] <- 0
    tem1 <- exp(- tem0^.alpha)
    y_hat <- .y2 %*% tem1/colSums(tem1)
    y_hat[is.na(y_hat)] <- 0
    return(t(y_hat))
  }
  gaussian_kernel <- function(.ep){
    tem0 <- exp(- (.ep*dis)/2)
    y_hat <- .y2 %*% tem0/colSums(tem0)
    return(t(y_hat))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] = 0
    y_hat <- .y2 %*% tem0/colSums(tem0)
    return(t(y_hat))
  }
  # Biweight
  biweight_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] = 0
    y_hat <- .y2 %*% tem0^2/colSums(tem0^2)
    y_hat[is.na(y_hat)] <- 0
    return(t(y_hat))
  }
  
  # Triweight
  triweight_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] = 0
    y_hat <- .y2 %*% tem0^3/colSums(tem0^3)
    y_hat[is.na(y_hat)] <- 0
    return(t(y_hat))
  }
  
  # Triangular
  triangular_kernel <- function(.ep){
    tem0 <- 1- .ep*dis
    tem0[tem0 < 0] <- 0
    y_hat <- .y2 %*% tem0/colSums(tem0)
    y_hat[is.na(y_hat)] = 0
    return(t(y_hat))
  }
  # Naive
  naive_kernel <- function(.ep){
    tem0 <- (.ep*dis < 1)
    y_hat <- .y2 %*% tem0/colSums(tem0)
    y_hat[is.na(y_hat)] = 0
    return(t(y_hat))
  }
  # Prediction
  kernel_list <- list(gaussian = gaussian_kernel,
                      epanechnikov = epanechnikov_kernel,
                      biweight = biweight_kernel,
                      triweight = triweight_kernel,
                      triangular = triangular_kernel,
                      naive = naive_kernel,
                      expo = expo_kernel,
                      c.expo = c.expo_kernel)
  res <- tibble(as.vector(kernel_list[[.kern]](.ep = epsilon)))
  names(res) <- ifelse(is.na(.meth), 
                       .kern, 
                       paste0(.kern, '_', .meth))
  return(res)
}

# ------------------------------------------------------------------------------------------------ #

# Functions: `predict_agg`
# ------------------------

### Prediction
predict_agg <- function(fitted_models,
                        new_data,
                        test_response = NULL){
  opt_param <- fitted_models$opt_parameters
  add_param <- fitted_models$add_parameters
  basic_mach <- fitted_models$basic_machines
  kern0 <- names(opt_param)
  kernel0 <- stringr::str_split(kern0, "_") %>%
    imap_dfc(.f = ~ tibble("{.y}" := .x)) 
  kerns <- kernel0[1,] %>%
    as.character
  opt_meths <- kernel0[2,] %>%
    as.character
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
          map_dfc(.f = (\(k) tibble('{{k}}' := FNN::knn.reg(train = mat_input[!basic_mach$id2,], 
                                                            test = mat_test, 
                                                            y = basic_mach$train_data$train_response[!basic_mach$id2],
                                                            k = built_models[[meth]][[k]])$pred)))
      }
      if(meth %in% c("tree", "rf")){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict(built_models[[meth]][[k]], df_test)))))
      }
      if(meth %in% c("lasso", "ridge")){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict.glmnet(built_models[[meth]][[k]], mat_test)))))
      }
      if(meth == "xgb"){
        pre <- 1:length(built_models[[meth]]) %>%
          map_dfc(.f = (\(k) tibble('{{k}}' := as.vector(predict(built_models[[meth]][[k]], mat_test)))))
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
  if(add_param$scale_machine){
    pred_test_all <- scale(pred_test_all, 
                           center = fitted_models$.scale$min,
                           scale = fitted_models$.scale$max - fitted_models$.scale$min)
  }
  # Prediction train2
  pred_train_all <- basic_mach$predict2
  colnames(pred_test_all) <- colnames(pred_train_all)
  d_train <- dim(pred_train_all)
  d_test <- dim(pred_test_all)
  pred_test_mat <- as.matrix(pred_test_all)
  pred_train_mat <- as.matrix(pred_train_all)
  # Distance matrix
  dist_mat <- function(kernel = "gaussian"){
    if(!(kernel %in% c("naive", "triangular"))){
      res_ <- 1:d_test[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums((pred_train_mat - matrix(rep(pred_test_mat[id,], 
                                                                                              d_train[1]), 
                                                                                          ncol = d_train[2], 
                                                                                          byrow = TRUE))^2)))))
    }
    if(kernel == "triangular"){
      res_ <- 1:d_test[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(abs(pred_train_mat - matrix(rep(pred_test_mat[id,], 
                                                                                                 d_train[1]), 
                                                                                             ncol = d_train[2], 
                                                                                             byrow = TRUE)))))))
    }
    if(kernel == "naive"){
      res_ <- 1:d_test[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(apply(abs(pred_train_mat - matrix(rep(pred_test_mat[id,], d_train[1]), 
                                                                                           ncol = d_train[2], 
                                                                                           byrow = TRUE)), 1, max)))))
    }
    return(res_)
  }
  dists <- 1:length(kerns) %>%
    map(.f = ~ dist_mat(kerns[.x]))
  tab_nam <- table(kerns)
  nam <- names(tab_nam[tab_nam > 1])
  vec <- rep(NA, length(kerns))
  for(id in nam){
    id_ <- kerns == id
    if(!is.null(id_)){
      vec[id_] = add_param$opt_methods[id_]
    }
  }
  prediction <- 1:length(kerns) %>% 
    map_dfc(.f = ~ kernel_pred(epsilon = opt_param[[kern0[.x]]]$opt_param,
                               .y2 = basic_mach$train_data$train_response[basic_mach$id2],
                               .distance = dists[[.x]],
                               .kern = kerns[.x], 
                               .sig = add_param$sig, 
                               .alp = add_param$alp,
                               .meth = vec[.x]))
  if(is.null(test_response)){
    return(list(fitted_aggregate = prediction,
                fitted_machine = pred_test_all))
  } else{
    error <- cbind(pred_test_all, prediction) %>%
      dplyr::mutate(y_test = test_response) %>%
      dplyr::summarise_all(.funs = ~ (. - y_test)) %>%
      dplyr::select(-y_test) %>%
      dplyr::summarise_all(.funs = ~ mean(.^2))
    return(list(fitted_aggregate = prediction,
                fitted_machine = pred_test0,
                mse = error))
  }
}

# ------------------------------------------------------------------------------------------------ #

# Function: `kernelAggReg`
# -----------------------

kernelAggReg <- function(train_design, 
                         train_response,
                         test_design,
                         test_response = NULL,
                         scale_input = FALSE,
                         scale_machine = FALSE,
                         build_machine = TRUE,
                         machines = NULL, 
                         splits = 0.5, 
                         n_cv = 5,
                         sig = 3,
                         alp = 2,
                         kernels = "gaussian",
                         optimizeMethod = "grad",
                         setBasicMachineParam = setBasicParameter(),
                         setGradParam = setGradParameter(),
                         setGridParam = setGridParameter()){
  # build machines + tune parameter
  fit_mod <- fit_parameter(train_design = train_design, 
                           train_response = train_response,
                           scale_input = scale_input,
                           scale_machine = scale_machine,
                           build_machine = build_machine,
                           machines = machines, 
                           splits = splits, 
                           n_cv = n_cv,
                           sig = sig,
                           alp = alp,
                           kernels = kernels,
                           optimizeMethod = optimizeMethod,
                           setBasicMachineParam = setBasicMachineParam,
                           setGradParam = setGradParam,
                           setGridParam = setGridParam)
  # prediction
  pred <- predict_agg(fitted_models = fit_mod,
                      new_data = test_design,
                      test_response = test_response)
  return(list(fitted_aggregate = pred$fitted_aggregate,
              fitted_machine = pred$fitted_machine,
              pred_train2 = fit_mod$basic_machines$predict2,
              opt_parameter = fit_mod$opt_parameters,
              mse = pred$mse,
              kernels = kernels,
              ind_D2 = fit_mod$basic_machines$id2))
}
# ------------------------------------------------------------------------------------------------ #

#### --------------------------------------------------------------------- ####
#### -------------------- MixCOBRA for regression ------------------------ ####
#### --------------------------------------------------------------------- ####




#### Check if package "pacman" is already installed 

lookup_packages <- installed.packages()[,1]
if(!("pacman" %in% lookup_packages)){
  install.packages("pacman")
}
  

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



# Basic Machine generator
# -----------------------
## Function: `setBasicParameter_Mix`
## -----------------------------
  
setBasicParameter_Mix <- function(lambda = NULL,
                              k = 5, 
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

#### --------------------------------------------------------------------- ####

## Function: `generateMachines_Mix`
## ----------------------------
  
generateMachines_Mix <- function(train_input, 
                             train_response,
                             scale_input = TRUE,
                             scale_machine = TRUE,
                             machines = NULL, 
                             splits = 0.5, 
                             basicMachineParam = setBasicParameter_Mix(),
                             silent = FALSE){
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
  maxs <- mins <- NULL
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
    mod <- tree(as.formula(paste("train_y1~", 
                                 paste(input_names, sep = "", collapse = "+"), 
                                 collapse = "", 
                                 sep = "")), 
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
  if(!silent){
    cat("\n* Building basic machines ...\n")
    cat("\t~ Progress:")
  }
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
    if(!silent){
      cat(" ... ", round(m/M, 2)*100L,"%", sep = "")
    }
  }
  max_M <- min_M <- NULL
  pred_D2_ <- pred_D2
  if(scale_machine){
    max_M <- map_dbl(.x = pred_D2, .f = max)
    min_M <- map_dbl(.x = pred_D2, .f = min)
    pred_D2 <- scale(pred_D2, center = min_M, scale = max_M - min_M)
  }
  return(list(fitted_remain = pred_D2,
              models = all_mod,
              id2 = !id_D1,
              train_data = list(train_input = train_input_scale, 
                                train_response = train_response,
                                predict_remain_org = pred_D2_,
                                min_machine = min_M,
                                max_machine = max_M,
                                min_input = mins,
                                max_input = maxs)))
}

#### --------------------------------------------------------------------- ####


# Optimization algorithm
# ----------------------
  
## Gradient descent algorithm
## --------------------------

### Function: `setGradParameter_Mix`
### ---------------------------
  
setGradParameter_Mix <- function(val_init = NULL,
                             rate = NULL, 
                             alpha_range = seq(0.0001, 10, length.out = 5),
                             beta_range = seq(0.1, 50, length.out = 5),
                             max_iter = 300, 
                             print_step = TRUE, 
                             print_result = TRUE,
                             figure = TRUE, 
                             coef_auto = c(0.1,0.1),
                             coef_log = 1,
                             coef_sqrt = 1,
                             coef_lm = 1,
                             deg_poly = 2,
                             base_exp = 1.5,
                             axes = c("alpha", "beta", "L1 norm of gradient"),
                             title = NULL,
                             threshold = 1e-10) {
  return(
    list(val_init = val_init,
         rate = rate,
         alpha_range = alpha_range,
         beta_range = beta_range,
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
         axes = axes,
         title = title,
         threshold = threshold
    )
  )
}

#### --------------------------------------------------------------------- ####

### Function: `gradOptimizer_Mix`
### -----------------------------

gradOptimizer_Mix <- function(obj_fun,
                          setParameter = setGradParameter_Mix(),
                          silent = FALSE) {
  start.time <- Sys.time()
  # Optimization step:
  # ==================
  spec_print <- function(x, dig = 5) return(ifelse(x > 1e-6, 
                                                   format(x, digit = dig, nsmall = dig), 
                                                   format(x, scientific = TRUE, digit = dig, nsmall = dig)))
  collect_val <- c()
  gradients <- c()
  if (is.null(setParameter$val_init)){
    range_alp <- rep(setParameter$alpha_range, length(setParameter$beta_range))
    range_bet <- rep(setParameter$beta_range, length(setParameter$alpha_range))
    tem <- map2_dbl(.x = range_alp,
                    .y = range_bet,
                    .f = ~ obj_fun(c(.x, .y)))
    id0 <- which.min(tem)
    val <- val0 <- c(range_alp[id0], range_bet[id0])
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
  if(setParameter$print_step & !silent){
    cat("\n* Gradient descent algorithm ...")
    cat("\n  Step\t|  alpha    ;  beta   \t|  Gradient (alpha ; beta)\t|  Threshold \n")
    cat(" ", rep("-", 80), sep = "")
    cat("\n   0 \t| ", spec_print(val0[1])," ; ", spec_print(val0[2]),
        "\t| ", spec_print(grad_[1], 6), " ; ", spec_print(grad_[2], 5), 
        " \t| ", setParameter$threshold, "\n")
    cat(" ", rep("-",80), sep = "")
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
  grad0 <- 10*grad_ 
  if (is.numeric(setParameter$rate) | rate_GD == "auto") {
    while (i < setParameter$max_iter) {
      if(any(is.na(grad_))){
        val0 <- c(runif(1, val0[1]*0.99, val0[1]*1.01), 
                  runif(1, val0[2]*0.99, val0[2]*1.01)) 
        grad_ = pracma::grad(
          f = obj_fun, 
          x0 = val0, 
          heps = .Machine$double.eps ^ (1 / 3)
        )
      }
      val <- val0 - lambda0 * grad_
      if (any(val < 0)){
        val[val < 0] <- val0[val < 0]/2
        lambda0[val < 0] <- lambda0[val < 0] / 2
      }
      if(i > 5){
        sign_ <- sign(grad_) != sign(grad0)
        if(any(sign_)){
          lambda0[sign_] = lambda0[sign_]/2
        }
      }
      relative <- sum(abs(val - val0)) / sum(abs(val0))
      test_threshold <- max(relative, sum(abs(grad_ - grad0)))
      if (test_threshold > setParameter$threshold){
        val0 <- val
        grad0 <- grad_
      } else{
        break
      }
      grad_ <- pracma::grad(
        f = obj_fun, 
        x0 = val0, 
        heps = .Machine$double.eps ^ (1 / 3)
      )
      i <- i + 1
      if(setParameter$print_step & !silent){
        cat("\n  ", i, "\t| ", spec_print(val[1], 4), " ; ", spec_print(val[2], 4), 
            "\t| ", spec_print(grad_[1], 5), " ; ", spec_print(grad_[2], 5), 
            "\t| ", test_threshold, "\r")
      }
      collect_val <- rbind(collect_val, val)
      gradients <- rbind(gradients, grad_)
    }
  }
  else{
    while (i < setParameter$max_iter) {
      if(any(is.na(grad_))){
        val0 <- c(runif(1, val0[1]*0.99, val0[1]*1.01), 
                  runif(1, val0[2]*0.99, val0[2]*1.01)) 
        grad_ = pracma::grad(
          f = obj_fun, 
          x0 = val0, 
          heps = .Machine$double.eps ^ (1 / 3)
        )
      }
      val <- val0 - lambda0(i) * grad_
      if (any(val < 0)){
        val[val < 0] <- val0[val < 0]/2
        r0[val < 0] <- r0[val < 0] / 2
      }
      if(i > 5){
        sign_ <- sign(grad_) != sign(grad0)
        if(any(sign_)){
          r0[sign_] <- r0[sign_] / 2
        }
      }
      relative <- sum(abs(val - val0)) / sum(abs(val0))
      test_threshold <- max(relative, sum(abs(grad_ - grad0)))
      if (test_threshold > setParameter$threshold){
        val0 <- val
        grad0 <- grad_
      }else{
        break
      }
      grad_ <- pracma::grad(
        f = obj_fun, 
        x0 = val0, 
        heps = .Machine$double.eps ^ (1 / 3)
      )
      if(setParameter$print_step & !silent){
        cat("\n  ", i, "\t| ", spec_print(val[1], 4), " ; ", spec_print(val[2], 4), 
            "\t| ", spec_print(grad_[1], 5), " ; ", spec_print(grad_[2], 5), 
            "\t| ", test_threshold, "\r")
      }
      i <- i + 1
      collect_val <- rbind(collect_val, val)
      gradients <- rbind(gradients, grad_)
    }
  }
  opt_ep <- val
  opt_risk <- obj_fun(opt_ep)
  if(setParameter$print_step & !silent){
    cat(rep("-", 80), sep = "")
    if(sum(abs(grad_)) == 0){
      cat("\n Stopped| ", spec_print(val[1], 4), " ; ", spec_print(val[2], 4), 
          "\t|\t ", 0, 
          "\t\t| ", test_threshold)
    }else{
      cat("\n Stopped| ", spec_print(val[1], 4), " ; ", spec_print(val[2], 4), 
          "\t| ", spec_print(grad_[1]), " ; ", spec_print(grad_[2]), 
          "\t| ", test_threshold)
    } 
  }
  if(setParameter$print_result & !silent){
    cat("\n ~ Observed parameter: (alpha, beta) = (", opt_ep[1], ", ", opt_ep[2], ") in",i, "itertaions.")
  }
  if (setParameter$figure) {
    if(is.null(setParameter$title)){
      tit <- paste("<b> L1 norm of gradient as a function of</b> (",
                   setParameter$axes[1],",", 
                   setParameter$axes[2], 
                   ")")
    } else{
      tit <- setParameter$title
    }
    siz = length(collect_val[,1])
    fig <- tibble(x = collect_val[,1],
                  y = collect_val[,2],
                  z = apply(abs(gradients), 1, sum)) %>%
      plot_ly(x = ~x, y = ~y) %>% 
      add_trace(z = ~z,
                type = "scatter3d",
                mode = "lines",
                line = list(width = 6, 
                            color = ~z, 
                            colorscale = 'Viridis'),
                name = "Gradient step") %>%
      add_trace(x = c(opt_ep[1], opt_ep[1]),
                y = c(0, opt_ep[2]),
                z = ~c(z[siz], z[siz]),
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
                z = ~c(z[siz], z[siz]),
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
                z = ~z[siz],
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
             scene = list(
               xaxis = list(title = setParameter$axes[1]),
               yaxis = list(title = setParameter$axes[2]),
               zaxis = list( title = setParameter$axes[3])))
    fig %>% print
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

#### --------------------------------------------------------------------- ####

## Grid search algorithm
## ---------------------

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
              run.time = difftime(t1, 
                                  t0, 
                                  units = "secs")[[1]])
  )
}

#### --------------------------------------------------------------------- ####

## Function: `dist_matrix_Mix`
## -----------------------

dist_matrix_Mix <- function(basicMachines,
                        n_cv = 5,
                        kernel = "gausian",
                        id_shuffle = NULL){
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
  L1 <- 1:n_cv %>%
    map(.f = (\(x) pair_dist(df_input[shuffle != x,],
                             df_input[shuffle == x,])))
  L2 <- 1:n_cv %>%
    map(.f = (\(x) pair_dist(df_mach[shuffle != x,],
                             df_mach[shuffle == x,])))
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
                          scale_machine = TRUE,
                          splits = 0.5, 
                          n_cv = 5,
                          inv_sigma = sqrt(.5),
                          alp = 2,
                          kernels = "gaussian",
                          optimizeMethod = "grad",
                          setBasicMachineParam = setBasicParameter_Mix(),
                          setGradParam = setGradParameter_Mix(),
                          setGridParam = setGridParameter_Mix(),
                          silent = FALSE){
  kernels_lookup <- c("gaussian", "epanechnikov", "biweight", "triweight", "triangular", "naive")
  kernel_real <- kernels %>%
    sapply(FUN = function(x) return(match.arg(x, kernels_lookup)))
  if(is.null(train_predictions)){
    mach2 <- generateMachines_Mix(train_input = train_input,
                              train_response = train_response,
                              scale_input = scale_input,
                              scale_machine = scale_machine,
                              machines = machines,
                              splits = splits,
                              basicMachineParam = setBasicMachineParam,
                              silent = silent)
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
    if(scale_machine){
      min_ <- map_dbl(train_predictions, .f = min)
      max_ <- map_dbl(train_predictions, .f = max)
      mach2$train_data$min_machine = min_
      mach2$train_data$max_amchine = max_
      mach2$fitted_remain <- scale(train_predictions, 
                                   center = min_, 
                                   scale = max_ - min_)
    }
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
  for (k_ in 1:n_ker){
    ker <- kernel_real[k_]
    if(ker == "naive"){
      dist_all[["naive"]] <- dist_matrix_Mix(basicMachines = mach2,
                                         n_cv = n_cv,
                                         kernel = "naive",
                                         id_shuffle = id_shuf)
    } else{
      if(ker == "triangular"){
        dist_all[["triangular"]] <- dist_matrix_Mix(basicMachines = mach2,
                                                n_cv = n_cv,
                                                kernel = "triangular",
                                                id_shuffle = id_shuf)
      } else{
        if(if_euclid){
          dist_all[[ker]] <- dist_all[[id_euclid]]
        } else{
          dist_all[[ker]] <- dist_matrix_Mix(basicMachines = mach2,
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
  # Gaussian
  gaussian_kernel <- function(.ep,
                              .dist_matrix,
                              .train_response2,
                              .inv_sigma = inv_sigma,
                              .alpha = alp){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(exp(- (x[1]*D1+x[2]*D2)^(.alpha/2)*.inv_sigma^.alpha))
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = .dist_matrix$dist_machine[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(.ep,
                                  .dist_matrix,
                                  .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D))
      tem0[tem0 < 0] = 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = .dist_matrix$dist_machine[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Biweight
  biweight_kernel <- function(.ep,
                              .dist_matrix,
                              .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] = 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0^2/colSums(tem0^2)
      y_hat[is.na(y_hat)] <- 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = .dist_matrix$dist_machine[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Triweight
  triweight_kernel <- function(.ep,
                               .dist_matrix,
                               .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] = 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0^3/colSums(tem0^3)
      y_hat[is.na(y_hat)] <- 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = .dist_matrix$dist_machine[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Triangular
  triangular_kernel <- function(.ep,
                                .dist_matrix,
                                .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- as.matrix(1- (x[1]*D1+x[2]*D2))
      tem0[tem0 < 0] <- 0
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      y_hat[is.na(y_hat)] = 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = .dist_matrix$dist_machine[[.x]]))
    return(Reduce("+", temp))
  }
  
  # Naive
  naive_kernel <- function(.ep,
                           .dist_matrix,
                           .train_response2){
    kern_fun <- function(x, id, D1, D2){
      tem0 <- (as.matrix((x[1]*D1+x[2]*D2)) < 1)
      y_hat <- .train_response2[.dist_matrix$id_shuffle != id] %*% tem0/colSums(tem0)
      y_hat[is.na(y_hat)] = 0
      return(sum((y_hat - .train_response2[.dist_matrix$id_shuffle == id])^2))
    }
    temp <- map(.x = 1:.dist_matrix$n_cv, 
                .f = ~ kern_fun(x = .ep, 
                                id = .x,
                                D1 = .dist_matrix$dist_input[[.x]], 
                                D2 = .dist_matrix$dist_machine[[.x]]))
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
  names(error_func) <- kernels
  # list of prameter setup
  list_param <- list(grad = setGradParam,
                     GD = setGradParam,
                     grid = setGridParam)
  # list of optimizer
  list_optimizer <- list(grad = gradOptimizer_Mix,
                         GD = gradOptimizer_Mix,
                         grid = gridOptimizer_Mix)
  optMethods <- optimizeMethod
  if(length(kernels) != length(optMethods)){
    warning("* kernels and optimization methods differ in sides! Grid search will be used!")
    optMethods = rep("grid", length(kernels))
  }
  
  # Optimization
  parameters <- map2(.x = kernels,
                     .y = optMethods, 
                     .f = ~ list_optimizer[[.y]](obj_fun = error_func[[.x]],
                                                 setParameter = list_param[[.y]],
                                                 silent = silent))
  names(parameters) <- paste0(kernel_real, "_", optMethods)
  return(list(opt_parameters = parameters,
              add_parameters = list(inv_sigma = inv_sigma,
                                    alp = alp,
                                    opt_methods = optimizeMethod),
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
                        .alp = 2,
                        .meth = NA){
  distD <- as.matrix(theta[1]*.dist1+theta[2]*.dist1)
  # Kernel functions
  # ================
  gaussian_kernel <- function(D,
                              .inv_sigma = .inv_sig,
                              .alpha = .alp){
    tem0 <- exp(- D^(.alpha/2)*.inv_sig^.alpha)
    y_hat <- .y2 %*% tem0/colSums(tem0)
    return(t(y_hat))
  }
  
  # Epanechnikov
  epanechnikov_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] = 0
    y_hat <- .y2 %*% tem0/colSums(tem0)
    return(t(y_hat))
  }
  # Biweight
  biweight_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] = 0
    y_hat <- .y2 %*% tem0^2/colSums(tem0^2)
    y_hat[is.na(y_hat)] <- 0
    return(t(y_hat))
  }
  
  # Triweight
  triweight_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] = 0
    y_hat <- .y2 %*% tem0^3/colSums(tem0^3)
    y_hat[is.na(y_hat)] <- 0
    return(t(y_hat))
  }
  
  # Triangular
  triangular_kernel <- function(D){
    tem0 <- 1- D
    tem0[tem0 < 0] <- 0
    y_hat <- .y2 %*% tem0/colSums(tem0)
    y_hat[is.na(y_hat)] = 0
    return(t(y_hat))
  }
  # Naive
  naive_kernel <- function(D){
    tem0 <- (D < 1)
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
                      naive = naive_kernel)
  res <- tibble(as.vector(kernel_list[[.kern]](D = distD)))
  names(res) <- ifelse(is.na(.meth), 
                       .kern, 
                       paste0(.kern, '_', .meth))
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
  kernel0 <- stringr::str_split(kern0, "_") %>%
    imap_dfc(.f = ~ tibble("{.y}" := .x)) 
  kerns <- kernel0[1,] %>%
    as.character
  opt_meths <- kernel0[2,] %>%
    as.character
  new_data_ <- new_data
  mat_input <- as.matrix(basic_mach$train_data$train_input)
  if(!is.null(basic_mach$train_data$min_input)){
    new_data_ <- scale(new_data, 
                       center = basic_mach$train_data$min_input, 
                       scale = basic_mach$train_data$max_input - basic_mach$train_data$min_input)
  }
  if(is.matrix(new_data_)){
    mat_test <- new_data_
    df_test <- as_tibble(new_data_)
  } else {
    mat_test <- as.matrix(new_data_)
    df_test <- new_data_
  }
  if(is.null(basic_mach$models)){
    pred_test_all <- new_pred
    pred_test0 <- new_pred
  } else{
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
      names(pre) <- names(built_models[[meth]])
      return(pre)
    }
    pred_test_all <- names(built_models) %>%
      map_dfc(.f = pred_test)
    pred_test0 <- pred_test_all
  }
  if(!is.null(basic_mach$train_data$min_machine)){
    pred_test_all <- scale(pred_test0, 
                           center = basic_mach$train_data$min_machine,
                           scale = basic_mach$train_data$max_machine - basic_mach$train_data$min_machine)
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
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums((mat_input[basic_mach$id2,] - matrix(rep(new_data_[id,], 
                                                                                                          d_train_input[1]), 
                                                                                                      ncol = d_train_input[2], 
                                                                                                      byrow = TRUE))^2)))))
      res_2 <- 1:d_test[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums((pred_train_mat - matrix(rep(pred_test_mat[id,], 
                                                                                              d_train[1]), 
                                                                                          ncol = d_train[2], 
                                                                                          byrow = TRUE))^2)))))
    }
    if(kernel == "triangular"){
      res_1 <- 1:d_test_input[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(abs(mat_input[basic_mach$id2,] - matrix(rep(new_data_[id,], 
                                                                                                             d_train_input[1]), 
                                                                                                         ncol = d_train_input[2], 
                                                                                                         byrow = TRUE)))))))
      res_2 <- 1:d_test[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(rowSums(abs(pred_train_mat - matrix(rep(pred_test_mat[id,], 
                                                                                                 d_train[1]), 
                                                                                             ncol = d_train[2], 
                                                                                             byrow = TRUE)))))))
    }
    if(kernel == "naive"){
      res_1 <- 1:d_test_input[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(apply(abs(mat_input[basic_mach$id2,] - matrix(rep(new_data_[id,], 
                                                                                                           d_train_input[1]),
                                                                                                       ncol = d_train_input[2], 
                                                                                                       byrow = TRUE)), 1, max)))))
      res_2 <- 1:d_test[1] %>%
        map_dfc(.f = (\(id) tibble('{{id}}' := as.vector(apply(abs(pred_train_mat - matrix(rep(pred_test_mat[id,], d_train[1]), 
                                                                                           ncol = d_train[2], 
                                                                                           byrow = TRUE)), 1, max)))))
    }
    return(list(dist_input = res_1,
                dist_machine = res_2))
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
    map_dfc(.f = ~ kernel_pred_Mix(theta = opt_param[[kern0[.x]]]$opt_param,
                                   .y2 = basic_mach$train_data$train_response[basic_mach$id2],
                                   .dist1 = dists[[.x]]$dist_input,
                                   .dist2 = dists[[.x]]$dist_machine,
                                   .kern = kerns[.x], 
                                   .inv_sig = add_param$inv_sigma, 
                                   .alp = add_param$alp,
                                   .meth = vec[.x]))
  if(is.null(test_response)){
    return(list(fitted_aggregate = prediction,
                fitted_machine = pred_test0))
  } else{
    error <- cbind(pred_test0, prediction) %>%
      dplyr::mutate(y_test = test_response) %>%
      dplyr::summarise_all(.funs = ~ (. - y_test)) %>%
      dplyr::select(-y_test) %>%
      dplyr::summarise_all(.funs = ~ mean(.^2))
    return(list(fitted_aggregate = prediction,
                fitted_machine = pred_test0,
                mse = error))
  }
}

#### --------------------------------------------------------------------- ####

# Function : `MixCOBRARegressor` 
# ========================

MixCOBRARegressor <- function(train_input, 
                        train_response,
                        test_input,
                        train_predictions = NULL,
                        test_predictions = NULL,
                        test_response = NULL,
                        machines = NULL, 
                        scale_input = TRUE,
                        scale_machine = TRUE,
                        splits = 0.5, 
                        n_cv = 5,
                        inv_sigma = sqrt(.5),
                        alp = 2,
                        kernels = "gaussian",
                        optimizeMethod = "grad",
                        setBasicMachineParam = setBasicParameter_Mix(),
                        setGradParam = setGradParameter_Mix(),
                        setGridParam = setGridParameter_Mix(),
                        silent = FALSE){
  # build machines + tune parameter
  cat("\n\nMixCobra for regression\n-----------------------\n")
  fit_mod <- fit_parameter_Mix(train_input = train_input, 
                               train_response = train_response,
                               train_predictions = train_predictions,
                               machines = machines, 
                               scale_input = scale_input,
                               scale_machine = scale_machine,
                               splits = splits, 
                               n_cv = n_cv,
                               inv_sigma = inv_sigma,
                               alp = alp,
                               kernels = kernels,
                               optimizeMethod = optimizeMethod,
                               setBasicMachineParam = setBasicMachineParam,
                               setGradParam = setGradParam,
                               setGridParam = setGridParam,
                               silent = silent)
  # prediction
  pred <- predict_Mix(fitted_models = fit_mod,
                      new_data = test_input,
                      new_pred = test_predictions,
                      test_response = test_response)
  return(list(fitted_aggregate = pred$fitted_aggregate,
              fitted_machine = pred$fitted_machine,
              pred_train2 = fit_mod$basic_machines$fitted_remain,
              opt_parameter = fit_mod$opt_parameters,
              mse = pred$mse,
              kernels = kernels,
              ind_D2 = fit_mod$basic_machines$id2))
}

#### --------------------------------------------------------------------- ####

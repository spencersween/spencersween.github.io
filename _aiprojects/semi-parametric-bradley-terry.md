---
title: "Structural Deep Learning for Bradley Terry Models"
collection: aiprojects
permalink: /aiprojects/semi-parametric-bradley-terry/
excerpt: "A full R+Torch pipeline for semi-parametric Bradley–Terry models for LLM pairwise comparisons, using structural multi-head neural nets, automatic influence function construction, conditional inference, and policy learning."
author_profile: true
---

This tutorial walks through a full semi parametric Bradley Terry pipeline that combines:

- Neural networks for conditional preference and cost functions
- Structural information from the Bradley Terry design
- Judge-aware cross-fit estimation of neural nets
- Influence function representations for four target functionals
- Uniform confidence bands for all models
- Heterogeneity analysis in covariates and prompt space
- Policy trees that map prompts to model deployment rules

All code snippets are in R using `torch` for deep learning, `binsreg` for nonparametric heterogeneity, `policytree` for policy learning, and `fixest` for classical regression comparisons.

The pipeline is implemented on a preference dataset (`bt_comparia.csv`) where judges compare two models (or labs) on a given prompt, and ecologits provide energy usage for each model response.

---

## 1. Data Structure and Design Matrices

We begin from a CSV file that contains pairwise comparisons and covariates.

### 1.1 Variables

The key variables are:

- `Y_lab` and `Y_model`: binary outcomes equal to 1 if Creator A wins and 0 otherwise.
- `LD_*`, `MD_*`, `ED_*`: treatment indicators in `{-1, 0, +1}`.
  - `LD_*`: lab level Bradley Terry design.
  - `MD_*`: all model preferences.
  - `ED_*`: ecologits compatible model preferences (subset of `MD_*`).
- `X_*`: covariates for the prompt, such as log token length or embeddings.
- `did_vote`: indicator that the judge responded.
- `tied`: indicator that the judge declared a tie.
- `has_model_ecologits`: indicator that ecologits based energy data is available.
- `judge_id`: identifier for the judge.
- `log_energy_a`, `log_energy_b`: log energy usage for side A and side B respectively.

The core idea is to represent each comparison as:

- A scalar outcome `Y_i` indicating whether side A won.
- A vector of signed indicators `D_i` that encode which models were on side A or B.
- A vector of covariates `X_i` describing the prompt and context.

### 1.2 Data Loading and Preprocessing

A minimal data loading and preprocessing block looks like:

    rm(list = ls())
    gc()

    library(torch)
    library(coro)
    library(tidyverse)
    library(data.table)
    library(ggplot2)
    library(binsreg)
    library(policytree)
    library(fixest)

    set.seed(42)
    torch_manual_seed(42)

    device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

    csv_path <- "bt_comparia.csv"

    dat <- data.table::fread(csv_path) %>%
      as_tibble()

    drop_nres <- TRUE
    drop_ties <- TRUE
    use_labs  <- FALSE      # FALSE: work at model level
    use_ecol  <- TRUE       # TRUE: ecologits and energy costs available

    base_lab <- "LDopenai"
    base_mod <- "MDgpt_5"
    base_eco <- "EDgpt_5"

    dat_clean <- dat
    if (drop_nres) dat_clean <- dat_clean %>% filter(did_vote == 1)
    if (drop_ties) dat_clean <- dat_clean %>% filter(tied == 0)
    if (use_ecol)  dat_clean <- dat_clean %>% filter(has_model_ecologits == 1)

We select which Bradley Terry family to use based on whether we work with labs, all models, or the ecologits subset.

    if (use_labs) {
      base_name <- base_lab
      prefix    <- "^LD"
      Y         <- as.matrix(dat_clean$Y_lab, ncol = 1)
    } else {
      if (use_ecol) {
        base_name <- base_eco
        prefix    <- "^ED"
        Y         <- as.matrix(dat_clean$Y_model, ncol = 1)
      } else {
        base_name <- base_mod
        prefix    <- "^MD"
        Y         <- as.matrix(dat_clean$Y_model, ncol = 1)
      }
    }

We now construct the preference and cost designs. The base model is used as the reference in the preference block and appears explicitly in the cost block.

    if (use_ecol && use_labs) {

      d_cols_pref <- c("LDanthropic", "LDcohere", "LDmistral", "LDgoogle")
      D_pref      <- as.matrix(dat_clean[, d_cols_pref, drop = FALSE])

      d_cols_cost <- c("LDanthropic", "LDcohere", "LDmistral", "LDgoogle", "LDopenai")
      D_cost      <- as.matrix(dat_clean[, d_cols_cost, drop = FALSE])

    } else {

      all_d_cols  <- grep(prefix, names(dat_clean), value = TRUE)
      d_cols_pref <- setdiff(all_d_cols, base_name)
      D_pref      <- as.matrix(dat_clean[, d_cols_pref, drop = FALSE])

      d_cols_cost <- c(base_name, d_cols_pref)
      D_cost      <- as.matrix(dat_clean[, d_cols_cost, drop = FALSE])
    }

Covariates and judge identifiers:

    x_cols <- grep("^X_", names(dat_clean), value = TRUE)
    X      <- as.matrix(dat_clean[, x_cols, drop = FALSE])

    Judge  <- as.matrix(dat_clean$judge_id, ncol = 1)

Energy and related quantities:

    if (use_ecol) {
      E <- cbind(dat_clean$log_energy_a, dat_clean$log_energy_b)
    } else {
      n_tmp <- nrow(dat_clean)
      E <- matrix(0, nrow = n_tmp, ncol = 2)
    }

Dimensions:

    nobs      <- nrow(D_pref)
    dimD_pref <- ncol(D_pref)
    dimD_cost <- ncol(D_cost)
    dimX      <- ncol(X)

    d_cols <- d_cols_pref

We use cross fitting over judges to debias the influence function construction. Splits are drawn at the judge level so that all comparisons from a single judge stay in the same fold.

    Nsplits <- 50L

    Splits <- tibble(Judge = Judge[, 1]) %>%
      group_by(Judge) %>%
      mutate(Splits = sample(1:Nsplits, size = 1)) %>%
      ungroup() %>%
      pull(Splits)

---

## 2. Torch Tensors and Cost Targets

We convert everything to torch tensors and create the cost outcomes implied by the ecologits design `D_cost`.

    X_torch      <- torch_tensor(X,      dtype = torch_float(), device = device)$view(c(nobs, dimX))
    D_pref_torch <- torch_tensor(D_pref, dtype = torch_float(), device = device)$view(c(nobs, dimD_pref))
    D_cost_torch <- torch_tensor(D_cost, dtype = torch_float(), device = device)$view(c(nobs, dimD_cost))
    Y_torch      <- torch_tensor(Y,      dtype = torch_float(), device = device)$view(c(nobs, 1L))
    Judge_torch  <- torch_tensor(Judge,  dtype = torch_float(), device = device)$view(c(nobs, 1L))
    E_torch      <- torch_tensor(E,      dtype = torch_float(), device = device)$view(c(nobs, 2L))

    Y_vec <- Y_torch$view(c(nobs))

For costs we have a design where, for each model and comparison, the sign indicates whether the model is on side A or B. We construct observed costs and masks.

    mask_pos_cost      <- (D_cost_torch ==  1)
    mask_neg_cost      <- (D_cost_torch == -1)
    mask_active_torch  <- (mask_pos_cost | mask_neg_cost)$to(dtype = torch_float())

    E_A <- E_torch[, 1]$unsqueeze(2)$expand(c(nobs, dimD_cost))
    E_B <- E_torch[, 2]$unsqueeze(2)$expand(c(nobs, dimD_cost))

    Cost_target_torch <- mask_pos_cost$to(dtype = torch_float()) * E_A +
      mask_neg_cost$to(dtype = torch_float()) * E_B

The entries of `Cost_target_torch` are observed only when `mask_active_torch` equals one.

---

## 3. Hyperparameters, Regularization, and Activation

The semi parametric pipeline relies on multiple networks:

- `ThetaNet`: conditional preference parameters for non base models.
- `HessianLearner`: conditional Hessian for the preference block.
- `CostNet`: conditional energy cost for all models.
- `PropensityNet`: conditional probability of observing a cost for each model.

We parameterize their capacity and regularization via a shared set of hyperparameters.

    patience_theta    <- 10L
    patience_hessian  <- 10L
    min_delta_theta   <- 1e-5
    min_delta_hessian <- 1e-5

    theta_hidden_dims      <- c(20, 20)
    theta_activation       <- "leaky"
    theta_final_activation <- "identity"
    theta_dropout          <- 0.01
    theta_batch_norm       <- FALSE
    theta_lr               <- 1e-3
    theta_num_epochs       <- 1000L
    theta_batch_size       <- 10000L
    theta_clamp_val        <- log((1 - 1e-5) / (1e-5))

    hessian_hidden_dims <- c(20, 20)
    hessian_activation  <- "leaky"
    hessian_dropout     <- 0.01
    hessian_batch_norm  <- FALSE
    hessian_lr          <- 1e-3
    hessian_num_epochs  <- 1000L
    hessian_batch_size  <- 10000L
    hessian_max_eig     <- 0.25

    cost_hidden_dims      <- c(20, 20)
    cost_activation       <- "relu"
    cost_final_activation <- "leaky"
    cost_dropout          <- 0.01
    cost_batch_norm       <- FALSE
    cost_lr               <- 1e-3
    cost_num_epochs       <- 1000L
    cost_batch_size       <- 10000L

    prop_hidden_dims      <- c(20, 20)
    prop_activation       <- "leaky"
    prop_final_activation <- "identity"
    prop_dropout          <- 0.01
    prop_batch_norm       <- FALSE
    prop_lr               <- 1e-3
    prop_num_epochs       <- 1000L
    prop_batch_size       <- 10000L
    prop_logit_clamp      <- log((1 - 1e-5) / (1e-5))

    alpha   <- 0.05
    z_point <- qnorm(1 - alpha / 2)

We use both L2 and L1 regularization, and learning rate schedulers.

    theta_weight_decay   <- 1e-5
    hessian_weight_decay <- 1e-5
    cost_weight_decay    <- 1e-5
    prop_weight_decay    <- 1e-5

    theta_l1_lambda   <- 1e-5
    hessian_l1_lambda <- 1e-5
    cost_l1_lambda    <- 1e-5
    prop_l1_lambda    <- 1e-5

    theta_scheduler_type   <- "cosine"
    hessian_scheduler_type <- "cosine"
    cost_scheduler_type    <- "cosine"
    prop_scheduler_type    <- "cosine"

    theta_scheduler_params <- list(
      step_size        = 20L,
      gamma            = 0.1,
      T_max            = theta_num_epochs,
      plateau_factor   = 0.1,
      plateau_patience = 10L
    )

    hessian_scheduler_params <- list(
      step_size        = 20L,
      gamma            = 0.1,
      T_max            = hessian_num_epochs,
      plateau_factor   = 0.1,
      plateau_patience = 10L
    )

    cost_scheduler_params <- list(
      step_size        = 20L,
      gamma            = 0.1,
      T_max            = cost_num_epochs,
      plateau_factor   = 0.1,
      plateau_patience = 10L
    )

    prop_scheduler_params <- list(
      step_size        = 20L,
      gamma            = 0.1,
      T_max            = prop_num_epochs,
      plateau_factor   = 0.1,
      plateau_patience = 10L
    )

A small helper for activations:

    apply_activation <- function(x, act) {
      if (is.null(act) || act == "identity") {
        x
      } else if (act == "relu") {
        nnf_relu(x)
      } else if (act == "leaky") {
        nnf_leaky_relu(x)
      } else if (act == "elu") {
        nnf_elu(x)
      } else if (act == "tanh") {
        torch_tanh(x)
      } else if (act == "sigmoid") {
        torch_sigmoid(x)
      } else {
        x
      }
    }

We also define L1 penalties and a scheduler factory.

    `%||%` <- function(a, b) if (is.null(a)) b else a

    add_l1_penalty <- function(loss, model, lambda_l1, device) {
      if (lambda_l1 <= 0) return(loss)
      l1 <- torch_zeros(1, dtype = torch_float(), device = device)
      for (p in model$parameters) {
        l1 <- l1 + p$abs()$sum()
      }
      loss + lambda_l1 * l1
    }

    make_scheduler <- function(optimizer, sched_type, params, num_epochs) {
      if (sched_type == "step") {
        step_size <- params$step_size %||% 50L
        gamma     <- params$gamma     %||% 0.1
        lr_step(optimizer, step_size = step_size, gamma = gamma)
      } else if (sched_type == "cosine") {
        T_max <- params$T_max %||% num_epochs
        lr_cosine_annealing(optimizer, T_max = T_max)
      } else if (sched_type == "plateau") {
        factor   <- params$plateau_factor   %||% 0.1
        patience <- params$plateau_patience %||% 10L
        lr_reduce_on_plateau(
          optimizer,
          factor   = factor,
          patience = patience,
          mode     = "min"
        )
      } else {
        NULL
      }
    }

---

## 4. Backbone MLP and Network Definitions

We use a shared `GenericMLP` as the backbone of all networks.

    GenericMLP <- nn_module(
      "GenericMLP",

      initialize = function(input_dim,
                            hidden_dims = c(64, 64),
                            activation  = "relu",
                            dropout     = 0,
                            batch_norm  = FALSE) {

        self$hidden_dims <- hidden_dims
        self$activation  <- activation
        self$dropout     <- dropout
        self$batch_norm  <- batch_norm

        self$layers <- nn_module_list()
        if (batch_norm) {
          self$bn_layers <- nn_module_list()
        }

        prev_dim <- input_dim
        for (h in hidden_dims) {
          self$layers$append(nn_linear(prev_dim, h))
          if (batch_norm) {
            self$bn_layers$append(nn_batch_norm1d(h))
          }
          prev_dim <- h
        }

        self$output_dim <- prev_dim
      },

      forward = function(x) {
        h <- x
        for (i in seq_along(self$hidden_dims)) {
          h <- self$layers[[i]](h)
          if (self$batch_norm) {
            h <- self$bn_layers[[i]](h)
          }
          if (self$activation == "relu") {
            h <- nnf_relu(h)
          } else if (self$activation == "leaky") {
            h <- nnf_leaky_relu(h)
          } else if (self$activation == "elu") {
            h <- nnf_elu(h)
          } else if (self$activation == "tanh") {
            h <- torch_tanh(h)
          }
          if (self$dropout > 0) {
            h <- nnf_dropout(h, p = self$dropout, training = self$training)
          }
        }
        h
      }
    )

### 4.1 ThetaNet: Conditional Bradley Terry Coefficients

The semi parametric Bradley Terry model assumes that for each comparison `i` and non base model `j` we have a coefficient `theta_j(X_i)`. Let `D_pref,i` be a vector of signed indicators for non base models. Then the log odds that side A wins is

$$
\eta_i = D_{\text{pref},i}^\top \theta(X_i), 
\quad 
p_i = \sigma(\eta_i).
$$

`ThetaNet` outputs the vector `theta(X)`.

    ThetaNet <- nn_module(
      "ThetaNet",

      initialize = function(input_dim,
                            n_items,
                            hidden_dims      = c(64, 64),
                            activation       = "relu",
                            final_activation = "identity",
                            dropout          = 0,
                            batch_norm       = FALSE,
                            clamp_val        = NULL) {

        self$n_items          <- n_items
        self$clamp_val        <- clamp_val
        self$final_activation <- final_activation

        self$backbone <- GenericMLP(
          input_dim   = input_dim,
          hidden_dims = hidden_dims,
          activation  = activation,
          dropout     = dropout,
          batch_norm  = batch_norm
        )

        self$out <- nn_linear(self$backbone$output_dim, n_items)
      },

      forward = function(x) {
        h <- self$backbone(x)
        theta_raw <- self$out(h)

        if (!is.null(self$clamp_val)) {
          cval <- self$clamp_val
          theta <- cval * torch_tanh(theta_raw / cval)
        } else {
          theta <- theta_raw
        }

        apply_activation(theta, self$final_activation)
      }
    )

### 4.2 HessianLearner: Conditional Preference Hessian

The per comparison log likelihood is
$$
\ell_i\big(\theta(X_i)\big) 
= Y_i \log p_i + (1 - Y_i)\log(1 - p_i).
$$

The gradient and Hessian with respect to the \(K\)-dimensional coefficient vector \(\theta(X_i)\) are
$$
g_i 
= \frac{\partial \ell_i}{\partial \theta(X_i)}
= (p_i - Y_i)\, D_{\text{pref},i},
$$
$$
H_i 
= \frac{\partial^2 \ell_i}{\partial \theta(X_i)\,\partial \theta(X_i)^\top}
= w_i\, D_{\text{pref},i} D_{\text{pref},i}^\top,
\quad
w_i = p_i(1 - p_i).
$$

Rather than using `H_i` directly, we train a Hessian net that learns `H_pref(X)` and enforces symmetric positive definiteness with eigenvalues bounded in `(0, hessian_max_eig]`.

    HessianLearner <- nn_module(
      "HessianLearner",

      initialize = function(input_dim,
                            n_items,
                            hidden_dims = c(64, 64),
                            activation  = "relu",
                            dropout     = 0,
                            batch_norm  = FALSE,
                            max_eig     = 0.25) {

        self$n_items <- n_items
        self$max_eig <- max_eig

        self$backbone <- GenericMLP(
          input_dim   = input_dim,
          hidden_dims = hidden_dims,
          activation  = activation,
          dropout     = dropout,
          batch_norm  = batch_norm
        )

        self$num_tril <- as.integer(n_items * (n_items + 1) / 2)
        self$out      <- nn_linear(self$backbone$output_dim, self$num_tril)

        tril_idx      <- torch_tril_indices(n_items, n_items, offset = 0)
        self$tril_row <- tril_idx[1, ]
        self$tril_col <- tril_idx[2, ]
      },

      forward = function(x) {
        B <- x$size(1)
        n <- self$n_items

        h <- self$backbone(x)
        z <- self$out(h)

        S <- torch_zeros(c(B, n, n), dtype = x$dtype, device = x$device)

        num_tril <- as.integer(self$tril_row$size(1))
        row_idx  <- (self$tril_row + 1L)$unsqueeze(1)$expand(c(B, num_tril))
        col_idx  <- (self$tril_col + 1L)$unsqueeze(1)$expand(c(B, num_tril))
        batch_idx <- torch_arange(
          start = 1, end = B, step = 1,
          dtype = torch_long(), device = x$device
        )$unsqueeze(2)$expand(c(B, num_tril))

        S$index_put_(
          indices = list(batch_idx, row_idx, col_idx),
          values  = z
        )

        S_t   <- S$transpose(-1, -2)
        diagS <- torch_diagonal(S, dim1 = -2, dim2 = -1)
        S     <- S + S_t - torch_diag_embed(diagS)

        eps_eig <- 1e-5
        eye_n   <- torch_eye(n, dtype = x$dtype, device = x$device)$unsqueeze(1)$expand(c(B, n, n))
        S       <- S + eps_eig * eye_n

        eig   <- linalg_eigh(S)
        evals <- eig[[1]]
        evecs <- eig[[2]]

        min_eig <- 1e-5
        max_eig <- self$max_eig

        mid   <- 0.5 * (max_eig + min_eig)
        range <- 0.5 * (max_eig - min_eig)

        eig_pos <- mid + range * torch_tanh(evals / mid)
        Lambda  <- torch_diag_embed(eig_pos)
        H       <- evecs$matmul(Lambda)$matmul(evecs$transpose(-1, -2))
        H
      }
    )

### 4.3 CostNet and PropensityNet

The cost block works with costs for all models, including the base. For each model \(k\), \(\kappa_k(X_i)\) approximates the log energy for that model on comparison \(i\). The empirical observations are
$$
C_{ik} =
\begin{cases}
E_{i1}, & \text{if } D_{\text{cost},ik} = +1,\\[4pt]
E_{i2}, & \text{if } D_{\text{cost},ik} = -1,\\[4pt]
\text{unobserved}, & \text{if } D_{\text{cost},ik} = 0.
\end{cases}
$$
Let `m_ik` be the indicator that model `k` is observed in comparison `i`. The squared loss is

$$
\ell_i^{\text{cost}}
= \frac{1}{2} \sum_k m_{ik} \big(\kappa_k(X_i) - C_{ik}\big)^2,
$$

with gradient

$$
g^{\text{cost}}_i 
= m_i \odot (\kappa_i - C_i).
$$

We model the Hessian via a diagonal block whose entries are propensities:

$$
H_{\text{cost}}(X) = \operatorname{diag}(\pi(X)),
\quad
\pi_k(X) = P(m_k = 1 \mid X).
$$

`CostNet` and `PropensityNet` implement these components.

    CostNet <- nn_module(
      "CostNet",

      initialize = function(input_dim,
                            n_items,
                            hidden_dims      = c(32, 32),
                            activation       = "leaky",
                            final_activation = "identity",
                            dropout          = 0,
                            batch_norm       = FALSE) {

        self$n_items          <- n_items
        self$final_activation <- final_activation

        self$backbone <- GenericMLP(
          input_dim   = input_dim,
          hidden_dims = hidden_dims,
          activation  = activation,
          dropout     = dropout,
          batch_norm  = batch_norm
        )

        self$out <- nn_linear(self$backbone$output_dim, n_items)
      },

      forward = function(x) {
        h <- self$backbone(x)
        kappa_raw <- self$out(h)
        apply_activation(kappa_raw, self$final_activation)
      }
    )

    PropensityNet <- nn_module(
      "PropensityNet",

      initialize = function(input_dim,
                            n_items,
                            hidden_dims      = c(32, 32),
                            activation       = "leaky",
                            final_activation = "identity",
                            dropout          = 0,
                            batch_norm       = FALSE,
                            logit_clamp      = NULL) {

        self$n_items          <- n_items
        self$logit_clamp      <- logit_clamp
        self$final_activation <- final_activation

        self$backbone <- GenericMLP(
          input_dim   = input_dim,
          hidden_dims = hidden_dims,
          activation  = activation,
          dropout     = dropout,
          batch_norm  = batch_norm
        )

        self$out <- nn_linear(self$backbone$output_dim, n_items)
      },

      forward = function(x) {
        h          <- self$backbone(x)
        logits_raw <- self$out(h)

        if (!is.null(self$logit_clamp)) {
          L <- self$logit_clamp
          logits_raw <- L * torch_tanh(logits_raw / L)
        }

        apply_activation(logits_raw, self$final_activation)
      }
    )

---

## 5. Dataset Wrappers for Dataloaders

We define simple dataset classes for the preference block, Hessian learning, cost outcomes, and propensities.

    bt_dataset <- dataset(
      name = "bt_dataset",

      initialize = function(X, D, y) {
        if (!inherits(X, "torch_tensor")) X <- torch_tensor(X)
        if (!inherits(D, "torch_tensor")) D <- torch_tensor(D)
        if (!inherits(y, "torch_tensor")) y <- torch_tensor(y)

        if (X$ndim == 1) X <- X$unsqueeze(2)
        if (D$ndim == 1) D <- D$unsqueeze(2)

        y <- y$view(c(-1))

        n_X <- as.integer(X$size(1))
        n_D <- as.integer(D$size(1))
        n_y <- as.integer(y$size(1))

        if (n_X != n_D) stop("bt_dataset: X and D lengths differ.")
        if (n_X != n_y) stop("bt_dataset: X and y lengths differ.")

        self$X <- X
        self$D <- D
        self$y <- y
      },

      .getitem = function(i) {
        list(
          X = self$X[i, ],
          D = self$D[i, ],
          y = self$y[i]
        )
      },

      .length = function() {
        as.integer(self$X$size(1))
      }
    )

    hessian_dataset <- dataset(
      name = "hessian_dataset",

      initialize = function(X, H_true) {
        if (!inherits(X, "torch_tensor"))      X      <- torch_tensor(X)
        if (!inherits(H_true, "torch_tensor")) H_true <- torch_tensor(H_true)

        if (X$ndim == 1) X <- X$unsqueeze(2)
        if (X$ndim != 2) stop("hessian_dataset: X must be 2D.")
        if (H_true$ndim != 3) stop("hessian_dataset: H_true must be 3D.")

        n_X <- as.integer(X$size(1))
        n_H <- as.integer(H_true$size(1))
        if (n_X != n_H) stop("hessian_dataset: X and H_true lengths differ.")

        self$X      <- X
        self$H_true <- H_true
      },

      .getitem = function(i) {
        list(
          X = self$X[i, ],
          H = self$H_true[i, , ]
        )
      },

      .length = function() {
        as.integer(self$X$size(1))
      }
    )

    cost_dataset <- dataset(
      name = "cost_dataset",

      initialize = function(X, Cost_target, mask_active) {
        if (!inherits(X, "torch_tensor"))           X           <- torch_tensor(X)
        if (!inherits(Cost_target, "torch_tensor")) Cost_target <- torch_tensor(Cost_target)
        if (!inherits(mask_active, "torch_tensor")) mask_active <- torch_tensor(mask_active)

        if (X$ndim == 1)           X           <- X$unsqueeze(2)
        if (Cost_target$ndim == 1) Cost_target <- Cost_target$unsqueeze(2)
        if (mask_active$ndim == 1) mask_active <- mask_active$unsqueeze(2)

        n_X  <- as.integer(X$size(1))
        n_Ct <- as.integer(Cost_target$size(1))
        n_m  <- as.integer(mask_active$size(1))
        if (n_X != n_Ct || n_X != n_m) stop("cost_dataset: incompatible lengths.")

        self$X           <- X
        self$Cost_target <- Cost_target
        self$mask_active <- mask_active
      },

      .getitem = function(i) {
        list(
          X           = self$X[i, ],
          Cost_target = self$Cost_target[i, ],
          mask_active = self$mask_active[i, ]
        )
      },

      .length = function() {
        as.integer(self$X$size(1))
      }
    )

    propensity_dataset <- dataset(
      name = "propensity_dataset",

      initialize = function(X, mask_active) {
        if (!inherits(X, "torch_tensor"))           X           <- torch_tensor(X)
        if (!inherits(mask_active, "torch_tensor")) mask_active <- torch_tensor(mask_active)

        if (X$ndim == 1)           X           <- X$unsqueeze(2)
        if (mask_active$ndim == 1) mask_active <- mask_active$unsqueeze(2)

        n_X <- as.integer(X$size(1))
        n_m <- as.integer(mask_active$size(1))
        if (n_X != n_m) stop("propensity_dataset: incompatible lengths.")

        self$X           <- X
        self$mask_active <- mask_active
      },

      .getitem = function(i) {
        list(
          X = self$X[i, ],
          m = self$mask_active[i, ]
        )
      },

      .length = function() {
        as.integer(self$X$size(1))
      }
    )

---

## 6. Cross Fitted Training Loop

We now describe the cross fitting loop. For each split:

1. Hold out one fold of judges for testing.
2. Within the remaining folds, carve out a validation subset for early stopping.
3. Train `ThetaNet` on train and validate on validation.
4. Compute true Hessians on train plus validation using `ThetaNet`.
5. Train `HessianLearner` to predict those Hessians from `X`.
6. Train `CostNet` and `PropensityNet` on cost data.
7. Use all networks to generate cross fitted predictions and gradients on the held out fold.

We allocate storage for cross fitted outputs:

    theta_all      <- torch_zeros(c(nobs, dimD_pref),              dtype = torch_float(), device = device)
    g_all          <- torch_zeros(c(nobs, dimD_pref),              dtype = torch_float(), device = device)
    H_hat_all      <- torch_zeros(c(nobs, dimD_pref, dimD_pref),   dtype = torch_float(), device = device)

    kappa_all      <- torch_zeros(c(nobs, dimD_cost),              dtype = torch_float(), device = device)
    g_cost_all     <- torch_zeros(c(nobs, dimD_cost),              dtype = torch_float(), device = device)
    pi_hat_all     <- torch_zeros(c(nobs, dimD_cost),              dtype = torch_float(), device = device)
    H_hat_cost_all <- torch_zeros(c(nobs, dimD_cost, dimD_cost),   dtype = torch_float(), device = device)

The outer loop:

    cat("Starting cross fitted training over", Nsplits, "splits.\n")

    for (s in seq_len(Nsplits)) {

      cat("\n============================\n")
      cat("Split", s, "of", Nsplits, "\n")

      idx_train_full <- which(Splits != s)
      idx_test       <- which(Splits == s)
      n_train_full   <- length(idx_train_full)

      if (n_train_full < 20) stop("Too few training obs in split ", s, ".")

      n_val   <- max(1L, floor(0.10 * n_train_full))
      idx_val <- sample(idx_train_full, n_val)
      idx_tr  <- setdiff(idx_train_full, idx_val)

      cat("  Train size:", length(idx_tr),
          "| Val size:", length(idx_val),
          "| Test size:", length(idx_test), "\n")

#### 6.1 Training ThetaNet

    train_ds_theta <- bt_dataset(
      X = X_torch[idx_tr, , drop = FALSE],
      D = D_pref_torch[idx_tr, , drop = FALSE],
      y = Y_vec[idx_tr]
    )

    val_ds_theta <- bt_dataset(
      X = X_torch[idx_val, , drop = FALSE],
      D = D_pref_torch[idx_val, , drop = FALSE],
      y = Y_vec[idx_val]
    )

    train_dl_theta <- dataloader(train_ds_theta, batch_size = theta_batch_size, shuffle = TRUE)
    val_dl_theta   <- dataloader(val_ds_theta,   batch_size = theta_batch_size, shuffle = FALSE)

    theta_model <- ThetaNet(
      input_dim        = dimX,
      n_items          = dimD_pref,
      hidden_dims      = theta_hidden_dims,
      activation       = theta_activation,
      final_activation = theta_final_activation,
      dropout          = theta_dropout,
      batch_norm       = theta_batch_norm,
      clamp_val        = theta_clamp_val
    )$to(device = device)

    optim_theta    <- optim_adamw(theta_model$parameters, lr = theta_lr, weight_decay = theta_weight_decay)
    theta_scheduler <- make_scheduler(optim_theta, theta_scheduler_type, theta_scheduler_params, num_epochs = theta_num_epochs)

    best_val_loss_theta <- Inf
    best_state_theta    <- NULL
    bad_epochs_theta    <- 0L

    for (epoch in seq_len(theta_num_epochs)) {
      theta_model$train()
      epoch_loss <- 0
      n_batches  <- 0L

      coro::loop(for (batch in train_dl_theta) {
        X_b <- batch$X$to(device = device)
        D_b <- batch$D$to(device = device)
        y_b <- batch$y
        if (!inherits(y_b, "torch_tensor")) y_b <- torch_tensor(y_b)
        y_b <- y_b$to(device = device)$view(c(-1))

        optim_theta$zero_grad()

        theta_b <- theta_model(X_b)
        eta_b   <- (D_b * theta_b)$sum(dim = 2)

        base_loss <- nnf_binary_cross_entropy_with_logits(eta_b, y_b, reduction = "mean")
        loss      <- add_l1_penalty(base_loss, theta_model, theta_l1_lambda, device)
        loss$backward()
        optim_theta$step()

        epoch_loss <- epoch_loss + loss$item()
        n_batches  <- n_batches + 1L
      })

      train_loss <- epoch_loss / max(1L, n_batches)

      theta_model$eval()
      val_loss_acc <- 0
      n_val_b      <- 0L

      with_no_grad({
        coro::loop(for (batch in val_dl_theta) {
          X_b <- batch$X$to(device = device)
          D_b <- batch$D$to(device = device)
          y_b <- batch$y
          if (!inherits(y_b, "torch_tensor")) y_b <- torch_tensor(y_b)
          y_b <- y_b$to(device = device)$view(c(-1))

          theta_b <- theta_model(X_b)
          eta_b   <- (D_b * theta_b)$sum(dim = 2)

          loss_b <- nnf_binary_cross_entropy_with_logits(eta_b, y_b, reduction = "mean")

          val_loss_acc <- val_loss_acc + loss_b$item()
          n_val_b      <- n_val_b + 1L
        })
      })

      val_loss <- val_loss_acc / max(1L, n_val_b)

      if (!is.null(theta_scheduler)) {
        if (theta_scheduler_type == "plateau") theta_scheduler$step(val_loss) else theta_scheduler$step()
      }

      cat(sprintf(
        "  [Split %d] Theta epoch %3d | train log10(loss)=%.6f | val log10(loss)=%.6f\n",
        s, epoch, log10(train_loss), log10(val_loss)
      ))

      if (val_loss < best_val_loss_theta - min_delta_theta) {
        best_val_loss_theta <- val_loss
        bad_epochs_theta    <- 0L
        best_state_theta    <- lapply(theta_model$state_dict(), function(x) x$clone())
      } else {
        bad_epochs_theta <- bad_epochs_theta + 1L
        if (bad_epochs_theta >= patience_theta) {
          cat("  Early stopping ThetaNet on split", s, "at epoch", epoch, "\n")
          break
        }
      }
    }

    if (!is.null(best_state_theta)) {
      theta_model$load_state_dict(best_state_theta)
    }
    theta_model$eval()

#### 6.2 Training HessianLearner

We use the true Hessians implied by the design and `ThetaNet` predictions as labels for `HessianLearner`.

    X_tv      <- X_torch[idx_train_full, , drop = FALSE]$to(device = device)
    D_pref_tv <- D_pref_torch[idx_train_full, , drop = FALSE]$to(device = device)
    Y_tv      <- torch_tensor(Y_vec[idx_train_full])$to(device = device)

    with_no_grad({
      theta_tv <- theta_model(X_tv)
      eta_tv   <- (D_pref_tv * theta_tv)$sum(dim = 2)
      p_tv     <- torch_sigmoid(eta_tv)
      w_tv     <- p_tv * (1 - p_tv)
      resid_tv <- p_tv - Y_tv

      D1 <- D_pref_tv$unsqueeze(3)
      D2 <- D_pref_tv$unsqueeze(2)

      H_true_tv <- w_tv$view(c(length(idx_train_full), 1, 1)) * (D1 * D2)
    })

    pos_tr  <- match(idx_tr,  idx_train_full)
    pos_val <- match(idx_val, idx_train_full)

    train_ds_H <- hessian_dataset(X = X_tv[pos_tr, , drop = FALSE],  H_true = H_true_tv[pos_tr, , ])
    val_ds_H   <- hessian_dataset(X = X_tv[pos_val, , drop = FALSE], H_true = H_true_tv[pos_val, , ])

    train_dl_H <- dataloader(train_ds_H, batch_size = hessian_batch_size, shuffle = TRUE)
    val_dl_H   <- dataloader(val_ds_H,   batch_size = hessian_batch_size, shuffle = FALSE)

    hessian_model <- HessianLearner(
      input_dim   = dimX,
      n_items     = dimD_pref,
      hidden_dims = hessian_hidden_dims,
      activation  = hessian_activation,
      dropout     = hessian_dropout,
      batch_norm  = hessian_batch_norm,
      max_eig     = hessian_max_eig
    )$to(device = device)

    optim_H        <- optim_adam(hessian_model$parameters, lr = hessian_lr, weight_decay = hessian_weight_decay)
    hessian_scheduler <- make_scheduler(optim_H, hessian_scheduler_type, hessian_scheduler_params, num_epochs = hessian_num_epochs)

    best_val_loss_H <- Inf
    best_state_H    <- NULL
    bad_epochs_H    <- 0L

    for (epoch in seq_len(hessian_num_epochs)) {
      hessian_model$train()
      epoch_loss <- 0
      n_batches  <- 0L

      coro::loop(for (batch in train_dl_H) {
        X_b <- batch$X$to(device = device)
        H_b <- batch$H$to(device = device)

        optim_H$zero_grad()

        H_pred <- hessian_model(X_b)
        base_loss <- nnf_mse_loss(H_pred, H_b)
        loss      <- add_l1_penalty(base_loss, hessian_model, hessian_l1_lambda, device)
        loss$backward()
        optim_H$step()

        epoch_loss <- epoch_loss + loss$item()
        n_batches  <- n_batches + 1L
      })

      train_loss <- epoch_loss / max(1L, n_batches)

      hessian_model$eval()
      val_loss_acc <- 0
      n_val_b      <- 0L

      with_no_grad({
        coro::loop(for (batch in val_dl_H) {
          X_b <- batch$X$to(device = device)
          H_b <- batch$H$to(device = device)

          H_pred <- hessian_model(X_b)
          loss_b <- nnf_mse_loss(H_pred, H_b)

          val_loss_acc <- val_loss_acc + loss_b$item()
          n_val_b      <- n_val_b + 1L
        })
      })

      val_loss <- val_loss_acc / max(1L, n_val_b)

      if (!is.null(hessian_scheduler)) {
        if (hessian_scheduler_type == "plateau") hessian_scheduler$step(val_loss) else hessian_scheduler$step()
      }

      cat(sprintf(
        "  [Split %d] Hessian pref epoch %3d | train log10(loss)=%.6f | val log10(loss)=%.6f\n",
        s, epoch, log10(train_loss), log10(val_loss)
      ))

      if (val_loss < best_val_loss_H - min_delta_hessian) {
        best_val_loss_H <- val_loss
        bad_epochs_H    <- 0L
        best_state_H    <- lapply(hessian_model$state_dict(), function(x) x$clone())
      } else {
        bad_epochs_H <- bad_epochs_H + 1L
        if (bad_epochs_H >= patience_hessian) {
          cat("  Early stopping HessianLearner (pref) on split", s, "at epoch", epoch, "\n")
          break
        }
      }
    }

    if (!is.null(best_state_H)) {
      hessian_model$load_state_dict(best_state_H)
    }
    hessian_model$eval()

#### 6.3 Cross Fitted Preference Predictions

On the held out fold we compute cross fitted `theta`, gradient, and Hessian.

    with_no_grad({
      X_te      <- X_torch[idx_test, , drop = FALSE]$to(device = device)
      D_pref_te <- D_pref_torch[idx_test, , drop = FALSE]$to(device = device)
      y_te      <- torch_tensor(Y_vec[idx_test])$to(device = device)

      theta_te <- theta_model(X_te)
      eta_te   <- (D_pref_te * theta_te)$sum(dim = 2)
      p_te     <- torch_sigmoid(eta_te)
      w_te     <- p_te * (1 - p_te)
      resid_te <- p_te - y_te

      g_te     <- resid_te$unsqueeze(2) * D_pref_te
      H_hat_te <- hessian_model(X_te)

      theta_all[idx_test, ]   <- theta_te$to(device = "cpu")
      g_all[idx_test, ]       <- g_te$to(device = "cpu")
      H_hat_all[idx_test, , ] <- H_hat_te$to(device = "cpu")
    })

#### 6.4 Training CostNet and PropensityNet

CostNet:

    train_ds_cost <- cost_dataset(
      X           = X_torch[idx_tr, , drop = FALSE],
      Cost_target = Cost_target_torch[idx_tr, , drop = FALSE],
      mask_active = mask_active_torch[idx_tr, , drop = FALSE]
    )

    val_ds_cost <- cost_dataset(
      X           = X_torch[idx_val, , drop = FALSE],
      Cost_target = Cost_target_torch[idx_val, , drop = FALSE],
      mask_active = mask_active_torch[idx_val, , drop = FALSE]
    )

    train_dl_cost <- dataloader(train_ds_cost, batch_size = cost_batch_size, shuffle = TRUE)
    val_dl_cost   <- dataloader(val_ds_cost,   batch_size = cost_batch_size, shuffle = FALSE)

    cost_model <- CostNet(
      input_dim        = dimX,
      n_items          = dimD_cost,
      hidden_dims      = cost_hidden_dims,
      activation       = cost_activation,
      final_activation = cost_final_activation,
      dropout          = cost_dropout,
      batch_norm       = cost_batch_norm
    )$to(device = device)

    optim_cost    <- optim_adam(cost_model$parameters, lr = cost_lr, weight_decay = cost_weight_decay)
    cost_scheduler <- make_scheduler(optim_cost, cost_scheduler_type, cost_scheduler_params, num_epochs = cost_num_epochs)

    best_val_cost   <- Inf
    best_state_cost <- NULL
    bad_epochs_cost <- 0L

    for (epoch in seq_len(cost_num_epochs)) {
      cost_model$train()
      train_loss_acc <- 0
      n_batches      <- 0L

      coro::loop(for (batch in train_dl_cost) {
        X_b  <- batch$X$to(device = device)
        Ct_b <- batch$Cost_target$to(device = device)
        m_b  <- batch$mask_active$to(device = device)

        optim_cost$zero_grad()

        kappa_b <- cost_model(X_b)
        res_b   <- (kappa_b - Ct_b) * m_b
        base_loss <- 0.5 * (res_b^2)$mean()
        loss      <- add_l1_penalty(base_loss, cost_model, cost_l1_lambda, device)
        loss$backward()
        optim_cost$step()

        train_loss_acc <- train_loss_acc + loss$item()
        n_batches      <- n_batches + 1L
      })

      train_loss <- train_loss_acc / max(1L, n_batches)

      cost_model$eval()
      val_loss_acc <- 0
      n_val_b      <- 0L

      with_no_grad({
        coro::loop(for (batch in val_dl_cost) {
          X_b  <- batch$X$to(device = device)
          Ct_b <- batch$Cost_target$to(device = device)
          m_b  <- batch$mask_active$to(device = device)

          kappa_b <- cost_model(X_b)
          res_b   <- (kappa_b - Ct_b) * m_b
          loss_b  <- 0.5 * (res_b^2)$mean()

          val_loss_acc <- val_loss_acc + loss_b$item()
          n_val_b      <- n_val_b + 1L
        })
      })

      val_loss <- val_loss_acc / max(1L, n_val_b)

      if (!is.null(cost_scheduler)) {
        if (cost_scheduler_type == "plateau") cost_scheduler$step(val_loss) else cost_scheduler$step()
      }

      cat(sprintf(
        "  [Split %d] Cost epoch %3d | train log10(MSE)=%.6f | val log10(MSE)=%.6f\n",
        s, epoch, log10(train_loss), log10(val_loss)
      ))

      if (val_loss < best_val_cost - min_delta_hessian) {
        best_val_cost   <- val_loss
        bad_epochs_cost <- 0L
        best_state_cost <- lapply(cost_model$state_dict(), function(x) x$clone())
      } else {
        bad_epochs_cost <- bad_epochs_cost + 1L
        if (bad_epochs_cost >= patience_hessian) {
          cat("  Early stopping CostNet on split", s, "at epoch", epoch, "\n")
          break
        }
      }
    }

    if (!is.null(best_state_cost)) {
      cost_model$load_state_dict(best_state_cost)
    }
    cost_model$eval()

PropensityNet:

    train_ds_prop <- propensity_dataset(
      X           = X_torch[idx_tr, , drop = FALSE],
      mask_active = mask_active_torch[idx_tr, , drop = FALSE]
    )

    val_ds_prop <- propensity_dataset(
      X           = X_torch[idx_val, , drop = FALSE],
      mask_active = mask_active_torch[idx_val, , drop = FALSE]
    )

    train_dl_prop <- dataloader(train_ds_prop, batch_size = prop_batch_size, shuffle = TRUE)
    val_dl_prop   <- dataloader(val_ds_prop,   batch_size = prop_batch_size, shuffle = FALSE)

    prop_model <- PropensityNet(
      input_dim        = dimX,
      n_items          = dimD_cost,
      hidden_dims      = prop_hidden_dims,
      activation       = prop_activation,
      final_activation = prop_final_activation,
      dropout          = prop_dropout,
      batch_norm       = prop_batch_norm,
      logit_clamp      = prop_logit_clamp
    )$to(device = device)

    optim_prop    <- optim_adam(prop_model$parameters, lr = prop_lr, weight_decay = prop_weight_decay)
    prop_scheduler <- make_scheduler(optim_prop, prop_scheduler_type, prop_scheduler_params, num_epochs = prop_num_epochs)

    best_val_prop   <- Inf
    best_state_prop <- NULL
    bad_epochs_prop <- 0L

    for (epoch in seq_len(prop_num_epochs)) {
      prop_model$train()
      train_loss_acc <- 0
      n_batches      <- 0L

      coro::loop(for (batch in train_dl_prop) {
        X_b <- batch$X$to(device = device)
        m_b <- batch$m$to(device = device)

        optim_prop$zero_grad()

        logits_b <- prop_model(X_b)
        base_loss <- nnf_binary_cross_entropy_with_logits(logits_b, m_b, reduction = "mean")
        loss      <- add_l1_penalty(base_loss, prop_model, prop_l1_lambda, device)
        loss$backward()
        optim_prop$step()

        train_loss_acc <- train_loss_acc + loss$item()
        n_batches      <- n_batches + 1L
      })

      train_loss <- train_loss_acc / max(1L, n_batches)

      prop_model$eval()
      val_loss_acc <- 0
      n_val_b      <- 0L

      with_no_grad({
        coro::loop(for (batch in val_dl_prop) {
          X_b <- batch$X$to(device = device)
          m_b <- batch$m$to(device = device)

          logits_b <- prop_model(X_b)
          loss_b   <- nnf_binary_cross_entropy_with_logits(logits_b, m_b, reduction = "mean")

          val_loss_acc <- val_loss_acc + loss_b$item()
          n_val_b      <- n_val_b + 1L
        })
      })

      val_loss <- val_loss_acc / max(1L, n_val_b)

      if (!is.null(prop_scheduler)) {
        if (prop_scheduler_type == "plateau") prop_scheduler$step(val_loss) else prop_scheduler$step()
      }

      cat(sprintf(
        "  [Split %d] Propensity epoch %3d | train log10(BCE)=%.6f | val log10(BCE)=%.6f\n",
        s, epoch, log10(train_loss), log10(val_loss)
      ))

      if (val_loss < best_val_prop - min_delta_hessian) {
        best_val_prop   <- val_loss
        bad_epochs_prop <- 0L
        best_state_prop <- lapply(prop_model$state_dict(), function(x) x$clone())
      } else {
        bad_epochs_prop <- bad_epochs_prop + 1L
        if (bad_epochs_prop >= patience_hessian) {
          cat("  Early stopping PropensityNet on split", s, "at epoch", epoch, "\n")
          break
        }
      }
    }

    if (!is.null(best_state_prop)) {
      prop_model$load_state_dict(best_state_prop)
    }
    prop_model$eval()

Cross fitted cost and propensity predictions on held out fold:

    with_no_grad({
      X_te_cf <- X_torch[idx_test, , drop = FALSE]$to(device = device)
      Ct_te   <- Cost_target_torch[idx_test, , drop = FALSE]$to(device = device)
      m_te    <- mask_active_torch[idx_test, , drop = FALSE]$to(device = device)

      kappa_te  <- cost_model(X_te_cf)
      res_cost  <- (kappa_te - Ct_te) * m_te
      g_cost_te <- res_cost

      logits_te <- prop_model(X_te_cf)
      pi_te     <- torch_sigmoid(logits_te)
      H_cost_te <- torch_diag_embed(pi_te)

      kappa_all[idx_test, ]        <- kappa_te$to(device = "cpu")
      g_cost_all[idx_test, ]       <- g_cost_te$to(device = "cpu")
      pi_hat_all[idx_test, ]       <- pi_te$to(device = "cpu")
      H_hat_cost_all[idx_test, , ] <- H_cost_te$to(device = "cpu")
    })

---

## 7. Structural Masking and Influence Functions

Before constructing influence functions, we use the structure of the design matrix to zero out entries of the conditional Hessian that are structurally zero.

    H_struct <- t(D_pref) %*% D_pref
    tol      <- 1e-12
    zero_mask_mat <- abs(H_struct) < tol
    zero_mask_torch <- torch_tensor(zero_mask_mat, dtype = torch_bool(), device = device)
    mask3 <- zero_mask_torch$unsqueeze(1)$expand(c(nobs, dimD_pref, dimD_pref))

    H_hat_all2 <- H_hat_all$masked_fill(mask3, 0)

We now define four influence function targets:

1. Preference parameters `theta_j(X)` for non base models.
2. Probabilities `P_j(X)` that each model is best (base plus non base).
3. Cost parameters `kappa_j(X)` for all models.
4. Cost times probability best, `f_j(X) = \kappa_j(X) P_j(X)` for all models.

### 7.1 Influence for Preference Parameters

We compute eigen based inverses of the Hessian and use

$$
\text{IF}_{\text{pref},i}
= \theta(X_i) - H_{\text{pref}}(X_i)^{-1}\, g_{\text{pref},i}.
$$

    n      <- nobs
    d_pref <- dimD_pref
    d_cost <- dimD_cost
    K_ext  <- d_cost

    g_vec <- g_all$unsqueeze(3)

    eps_val <- 0.01

    eig_pref   <- linalg_eigh(H_hat_all2)
    evals_pref <- eig_pref[[1]]
    evecs_pref <- eig_pref[[2]]

    evals_pref_clamped <- torch_clamp(evals_pref, min = eps_val, max = hessian_max_eig)

    Lambda_pref     <- torch_diag_embed(evals_pref_clamped)
    Lambda_pref_inv <- torch_diag_embed(1 / evals_pref_clamped)

    H_hat_all_clamp <- evecs_pref$matmul(Lambda_pref)$matmul(evecs_pref$transpose(-1, -2))
    H_inv_all       <- evecs_pref$matmul(Lambda_pref_inv)$matmul(evecs_pref$transpose(-1, -2))

    adj_pref <- H_inv_all$matmul(g_vec)$squeeze(3)
    IF_pref  <- theta_all - adj_pref

Sample means and standard errors:

    apply(as.matrix(IF_pref), 2, mean)
    apply(as.matrix(IF_pref), 2, sd) / sqrt(n)

### 7.2 Influence for Probabilities of Being Best

We extend `theta(X)` with a baseline logit of zero for the base model and pass it through a softmax:

$$
\text{logits}_i = \big(0, \theta(X_i)^\top\big)^\top, \quad P_i = \text{softmax}(\text{logits}_i).
$$

We obtain the Jacobian of `P` with respect to `theta` via autograd and use the delta method.

    theta_for_prob <- theta_all$detach()$clone()
    theta_for_prob$requires_grad_(TRUE)

    zero_col <- torch_zeros(c(n, 1L), dtype = theta_for_prob$dtype, device = theta_for_prob$device)
    logits_ext <- torch_cat(list(zero_col, theta_for_prob), dim = 2)
    P <- nnf_softmax(logits_ext, dim = 2)

    K_ext <- d_cost
    J_list <- vector("list", K_ext)

    for (k_idx in seq_len(K_ext)) {
      out_k <- P[, k_idx]$sum()
      grad_k <- autograd_grad(
        outputs      = list(out_k),
        inputs       = list(theta_for_prob),
        retain_graph = TRUE,
        create_graph = FALSE
      )[[1]]
      J_list[[k_idx]] <- grad_k$unsqueeze(2)
    }

    J <- torch_cat(J_list, dim = 2)

We then compute the influence of probabilities:

$$
\text{IF}_{\text{best},i}
= P_i - J_i\, H_{\text{pref}}(X_i)^{-1}\, g_{\text{pref},i}.
$$

    delta_theta <- H_inv_all$matmul(g_vec)
    IF_best     <- P - J$matmul(delta_theta)$squeeze(3)

    apply(as.matrix(IF_best), 2, mean)
    apply(as.matrix(IF_best), 2, sd) / sqrt(n)

### 7.3 Influence for Cost Parameters

The cost Hessian is modeled as `H_cost(X) = diag(pi(X))`, so its inverse is `diag(1 / pi(X))` up to clamping. The influence function is

$$
\text{IF}_{\text{cost},i}
= \kappa(X_i) - H_{\text{cost}}(X_i)^{-1}\, g^{\text{cost}}_i.
$$

    eps_cost <- 0.001

    pi_clamped      <- torch_clamp(pi_hat_all, min = eps_cost, max = 1 - eps_cost)
    H_inv_cost_all  <- torch_diag_embed(1 / pi_clamped)

    g_cost_vec <- g_cost_all$unsqueeze(3)
    adj_cost   <- H_inv_cost_all$matmul(g_cost_vec)$squeeze(3)
    IF_cost    <- kappa_all - adj_cost

    apply(as.matrix(IF_cost), 2, mean)
    apply(as.matrix(IF_cost), 2, sd) / sqrt(n)

### 7.4 Influence for Cost Times Probability Best

The fourth functional is
$$
f_j(X) = \kappa_j(X)\, P_j(X).
$$

We combine parameters into a single vector \(\psi = (\theta, \kappa)\) and define the block diagonal inverse Hessian:
$$
H_{\psi}^{-1}(X)
=
\begin{pmatrix}
H_{\text{pref}}^{-1}(X) & 0 \\[4pt]
0 & H_{\text{cost}}^{-1}(X)
\end{pmatrix}.
$$

The gradients of `f_j` with respect to `psi` are:

- With respect to `theta_l`: `∂f_j / ∂theta_l = kappa_j(X) * ∂P_j / ∂theta_l`.
- With respect to `kappa_l`: `∂f_j / ∂kappa_l = P_j(X)` when `l = j`, and zero otherwise.

We obtain:

    psi_all   <- torch_cat(list(theta_all, kappa_all), dim = 2)
    g_total   <- torch_cat(list(g_all, g_cost_all),    dim = 2)
    g_total_v <- g_total$unsqueeze(3)

    H_inv_big <- torch_zeros(c(n, d_pref + d_cost, d_pref + d_cost),
                              dtype = torch_float(), device = device)

    H_inv_big[, 1:d_pref, 1:d_pref] <- H_inv_all
    H_inv_big[, (d_pref + 1):(d_pref + d_cost), (d_pref + 1):(d_pref + d_cost)] <- H_inv_cost_all

    delta_psi_all <- H_inv_big$matmul(g_total_v)$squeeze(3)

Gradients with respect to `theta` and `kappa`:

    kappa_exp  <- kappa_all$unsqueeze(3)
    grad_theta <- kappa_exp * J

    grad_kappa <- torch_diag_embed(P)

    grad_psi <- torch_cat(list(grad_theta, grad_kappa), dim = 3)
    delta_exp <- delta_psi_all$unsqueeze(2)

    adj_cost_best <- (grad_psi * delta_exp)$sum(dim = 3)
    f_hat         <- kappa_all * P

    IF_cost_best <- f_hat - adj_cost_best

We are often interested in the ratio

$$
\theta_f = \mathbb{E}[f_j(X)],
\quad
\theta_p = \mathbb{E}[P_j(X)],
\quad
\theta_r = \frac{\theta_f}{\theta_p}.
$$

We use the delta method to get the influence for the ratio:

    theta_f_hat <- torch_mean(IF_cost_best, dim = 1)
    theta_p_hat <- torch_mean(IF_best,      dim = 1)

    theta_f_hat_b <- theta_f_hat$unsqueeze(1)
    theta_p_hat_b <- theta_p_hat$unsqueeze(1)

    IF_ratio <- IF_cost_best / theta_p_hat_b -
      (theta_f_hat_b / (theta_p_hat_b^2)) * (IF_best - theta_p_hat)

    IF_cost_best <- IF_ratio

    apply(as.matrix(IF_cost_best), 2, mean)
    apply(as.matrix(IF_cost_best), 2, sd) / sqrt(n)

---

## 8. Model Naming and Summary Tables

We map internal column names to human readable model names and construct tables for the four functionals.

    model_names_pref_internal <- d_cols_pref
    model_names_pref_short    <- sub("^LD|^MD|^ED", "", d_cols_pref)

    base_internal <- base_name
    base_short    <- sub("^LD|^MD|^ED", "", base_internal)

    model_names_cost_short <- c(base_short, model_names_pref_short)

### 8.1 Preference Parameters Table

    iff_pref <- as.matrix(IF_pref$to(device = "cpu"))

    n_pref <- nrow(iff_pref)
    d_p    <- ncol(iff_pref)

    est_pref <- colMeans(iff_pref)
    se_pref  <- apply(iff_pref, 2, sd) / sqrt(n_pref)
    z_pref   <- est_pref / se_pref

    tab_pref <- data.frame(
      Model    = model_names_pref_short,
      Estimate = est_pref,
      StdError = se_pref,
      Z        = z_pref,
      stringsAsFactors = FALSE
    )

    tab_pref <- tab_pref[order(tab_pref$Estimate), ]

    tab_pref <- tab_pref %>%
      mutate(
        Estimate = round(Estimate, 3),
        StdError = round(StdError, 3),
        Z        = round(Z, 3),
        Stars    = ifelse(abs(Z) > z_point & !is.na(Z), "*", "")
      )

    tab_pref

### 8.2 Probabilities of Being Best

    iff_best <- as.matrix(IF_best$to(device = "cpu"))

    n2   <- nrow(iff_best)
    K2   <- ncol(iff_best)

    est_best <- colMeans(iff_best)
    se_best  <- apply(iff_best, 2, sd) / sqrt(n2)
    z_best   <- est_best / se_best

    tab_best <- data.frame(
      Model    = model_names_cost_short,
      ProbBest = est_best,
      StdError = se_best,
      Z        = z_best,
      stringsAsFactors = FALSE
    )

    tab_best <- tab_best[order(tab_best$ProbBest, decreasing = TRUE), ]

    tab_best <- tab_best %>%
      mutate(
        ProbBest = round(ProbBest, 4),
        StdError = round(StdError, 4),
        Z        = round(Z, 3),
        Stars    = ifelse(abs(Z) > z_point & !is.na(Z), "*", "")
      )

    tab_best

### 8.3 Cost Parameters

    iff_cost <- as.matrix(IF_cost$to(device = "cpu"))

    n_c <- nrow(iff_cost)
    d_c <- ncol(iff_cost)

    est_cost <- colMeans(iff_cost)
    se_cost  <- apply(iff_cost, 2, sd) / sqrt(n_c)
    z_cost   <- est_cost / se_cost

    tab_cost <- data.frame(
      Model    = model_names_cost_short,
      Cost     = est_cost,
      StdError = se_cost,
      Z        = z_cost,
      stringsAsFactors = FALSE
    )

    tab_cost <- tab_cost[order(tab_cost$Cost), ]

    tab_cost <- tab_cost %>%
      mutate(
        Cost     = round(Cost, 4),
        StdError = round(StdError, 4),
        Z        = round(Z, 3),
        Stars    = ifelse(abs(Z) > z_point & !is.na(Z), "*", "")
      )

    tab_cost

### 8.4 Cost Times Probability Best

    iff_cost_best <- as.matrix(IF_cost_best$to(device = "cpu"))

    n_cb <- nrow(iff_cost_best)
    d_cb <- ncol(iff_cost_best)

    est_cost_best <- colMeans(iff_cost_best)
    se_cost_best  <- apply(iff_cost_best, 2, sd) / sqrt(n_cb)
    z_cost_best   <- est_cost_best / se_cost_best

    tab_cost_best <- data.frame(
      Model         = model_names_cost_short,
      CostTimesBest = est_cost_best,
      StdError      = se_cost_best,
      Z             = z_cost_best,
      stringsAsFactors = FALSE
    )

    tab_cost_best <- tab_cost_best[order(tab_cost_best$CostTimesBest), ]

    tab_cost_best <- tab_cost_best %>%
      mutate(
        CostTimesBest = round(CostTimesBest, 4),
        StdError      = round(StdError, 4),
        Z             = round(Z, 3),
        Stars         = ifelse(abs(Z) > z_point & !is.na(Z), "*", "")
      )

    tab_cost_best

---

## 9. Uniform Confidence Bands

Pointwise confidence intervals are easy to construct from the influence functions, but many applications require simultaneous bands across all models. We use a Gaussian multiplier bootstrap.

### 9.1 Multiplier Bootstrap

    set.seed(123)
    B <- 2000L

    bootstrap_uniform_crit <- function(iff_mat, est_vec, se_vec, B = 2000L, alpha = 0.05) {

      n  <- nrow(iff_mat)
      psi_centered <- sweep(iff_mat, 2, est_vec, FUN = "-")
      T_boot <- numeric(B)

      for (b in seq_len(B)) {
        xi <- rnorm(n)
        boot_score <- as.numeric(crossprod(xi, psi_centered)) / n
        Zb <- boot_score / se_vec
        T_boot[b] <- max(abs(Zb), na.rm = TRUE)
      }

      as.numeric(quantile(T_boot, probs = 1 - alpha, na.rm = TRUE))
    }

Uniform critical values for each functional:

    crit_theta_uni <- bootstrap_uniform_crit(iff_pref,       est_pref,       se_pref,       B = B, alpha = alpha)
    crit_best_uni  <- bootstrap_uniform_crit(iff_best,       est_best,       se_best,       B = B, alpha = alpha)
    crit_cost_uni  <- bootstrap_uniform_crit(iff_cost,       est_cost,       se_cost,       B = B, alpha = alpha)
    crit_cbest_uni <- bootstrap_uniform_crit(iff_cost_best,  est_cost_best,  se_cost_best,  B = B, alpha = alpha)

    cat("Uniform critical theta:",     round(crit_theta_uni, 3), "\n")
    cat("Uniform critical ProbBest:",  round(crit_best_uni, 3),  "\n")
    cat("Uniform critical cost:",      round(crit_cost_uni, 3),  "\n")
    cat("Uniform critical cost x best:", round(crit_cbest_uni, 3), "\n")

### 9.2 Confidence Band Plots

Preference parameters:

    tab_theta_band <- data.frame(
      Model    = model_names_pref_short,
      Estimate = est_pref,
      StdError = se_pref,
      stringsAsFactors = FALSE
    )

    tab_theta_band <- tab_theta_band[order(tab_theta_band$Estimate), ]

    tab_theta_band <- tab_theta_band %>%
      mutate(
        Lower = Estimate - crit_theta_uni * StdError,
        Upper = Estimate + crit_theta_uni * StdError
      )

    tab_theta_band$Model <- factor(tab_theta_band$Model, levels = tab_theta_band$Model)

    ggplot(tab_theta_band, aes(x = Model, y = Estimate)) +
      geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
      geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
      geom_point(size = 2) +
      coord_flip() +
      labs(
        x     = "Model",
        y     = "Bradley Terry preference parameter",
        title = "Uniform confidence bands for preference parameters"
      ) +
      theme_minimal(base_size = 12) +
      scale_y_continuous(n.breaks = 10, limits = c(-3, 3))

Analogous plots for probabilities, costs, and cost times best:

    tab_best_band <- data.frame(
      Model    = model_names_cost_short,
      Estimate = est_best,
      StdError = se_best,
      stringsAsFactors = FALSE
    ) %>%
      arrange(desc(Estimate)) %>%
      mutate(
        Lower = Estimate - crit_best_uni * StdError,
        Upper = Estimate + crit_best_uni * StdError,
        Model = factor(Model, levels = Model)
      )

    ggplot(tab_best_band, aes(x = Model, y = Estimate)) +
      geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
      geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
      geom_point(size = 2) +
      coord_flip() +
      labs(
        x     = "Model",
        y     = "Probability of being best",
        title = "Uniform confidence bands for probability best"
      ) +
      theme_minimal(base_size = 12)

    tab_cost_band <- data.frame(
      Model    = model_names_cost_short,
      Estimate = est_cost,
      StdError = se_cost,
      stringsAsFactors = FALSE
    ) %>%
      arrange(Estimate) %>%
      mutate(
        Lower = Estimate - crit_cost_uni * StdError,
        Upper = Estimate + crit_cost_uni * StdError,
        Model = factor(Model, levels = Model)
      )

    ggplot(tab_cost_band, aes(x = Model, y = Estimate)) +
      geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
      geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
      geom_point(size = 2) +
      coord_flip() +
      labs(
        x     = "Model",
        y     = "Energy cost parameter (log scale)",
        title = "Uniform confidence bands for cost parameters"
      ) +
      theme_minimal(base_size = 12)

    tab_cbest_band <- data.frame(
      Model    = model_names_cost_short,
      Estimate = est_cost_best,
      StdError = se_cost_best,
      stringsAsFactors = FALSE
    ) %>%
      arrange(Estimate) %>%
      mutate(
        Lower = Estimate - crit_cbest_uni * StdError,
        Upper = Estimate + crit_cbest_uni * StdError,
        Model = factor(Model, levels = Model)
      )

    ggplot(tab_cbest_band, aes(x = Model, y = Estimate)) +
      geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
      geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
      geom_point(size = 2) +
      coord_flip() +
      labs(
        x     = "Model",
        y     = "Cost times probability best",
        title = "Uniform confidence bands for cost times probability best"
      ) +
      theme_minimal(base_size = 12)

---

## 10. Heterogeneity in Token Length

A key benefit of the semi parametric approach is that influence functions can be analyzed as outcomes in follow up nonparametric regressions. For example, we can study how the probability that a specific model is best varies with log input token length.

### 10.1 Token Length Covariate

Assume the first covariate in `X` is `log_token_length`:

    token_length <- as.numeric(X[, 1])

We use `binsreg` to fit binned regressions of influence function entries on token length. First we set column names for the influence matrices to align with model names:

    colnames(iff_pref) <- model_names_pref_internal
    colnames(iff_best) <- model_names_cost_short

We then define a helper that runs `binsreg` and standardizes plot styling.

    plot_token_bins <- function(y_vec, main_title) {

      p_obj <- binsreg::binsreg(
        y          = y_vec,
        x          = token_length,
        dots       = c(1, 1),
        cb         = c(2, 2),
        line       = c(1, 1),
        plotxrange = c(0, log(8000)),
        nsims      = 2000,
        simsgrid   = 100,
        randcut    = 1
      )$bins_plot

      token_color <- "#D55E00"

      for (i in seq_along(p_obj$layers)) {
        g <- p_obj$layers[[i]]$geom
        if (inherits(g, "GeomRibbon")) {
          p_obj$layers[[i]]$aes_params$fill   <- token_color
          p_obj$layers[[i]]$aes_params$alpha  <- 0.5
        } else if (inherits(g, "GeomLine")) {
          p_obj$layers[[i]]$aes_params$colour    <- token_color
          p_obj$layers[[i]]$aes_params$linewidth <- 0.8
        } else if (inherits(g, "GeomPoint")) {
          p_obj$layers[[i]]$aes_params$colour <- token_color
        }
      }

      p_obj +
        geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5) +
        scale_y_continuous(n.breaks = 10) +
        scale_x_continuous(n.breaks = 10, limits = c(0, log(8000))) +
        ylab("Best model probability estimate and 95 percent UCB") +
        xlab("Log input message token length") +
        theme_bw() +
        ggtitle(main_title, subtitle = "Input token length heterogeneity")
    }

As an example, suppose the model named `mistral_medium_3_1` is contained in `model_names_cost_short`:

    iff_cond_token <- iff_best[, "mistral_medium_3_1"]
    plot_token_bins(iff_cond_token, "Conditional Bradley Terry - Mistral Medium 3.1")

This plot displays how the influence based estimates of the probability that Mistral Medium 3.1 is best vary as a function of log token length, including uniform confidence bands.

---

## 11. Prompt Specific Heterogeneity via Cosine Similarity

We can also study heterogeneity in prompt embedding space. Suppose we have a file of prompt embeddings:

- Each row corresponds to a comparison.
- Columns `e1`, `e2`, `...` correspond to embedding dimensions.

We load these embeddings and construct cosine similarities with a set of canonical tasks.

### 11.1 Loading Prompt Embeddings

    prompts <- read.csv(
      "prompt_embeddings.csv",
      stringsAsFactors = FALSE
    )

    prompt_mat <- as.matrix(prompts[, grep("^e[0-9]+$", names(prompts))])

We define three task types:

    task_names <- c("Coding Task", "Concept Explanation", "Text Generation")

We normalize rows to unit norm and compute cosine similarity between document embeddings (all comparisons) and task anchor vectors:

    row_normalize <- function(m) {
      norms <- sqrt(rowSums(m^2))
      norms[norms == 0] <- 1
      m / norms
    }

    doc_norm    <- row_normalize(X[, -1, drop = FALSE])
    prompt_norm <- row_normalize(prompt_mat)

    cosine_sim <- doc_norm %*% t(prompt_norm)
    colnames(cosine_sim) <- task_names

We focus again on a single model, say Mistral Medium 3.1, and construct a dataset for `binsreg` with multiple groups.

    iff_plot  <- iff_best[, "mistral_medium_3_1", drop = TRUE]
    n_plot    <- length(iff_plot)

    dataset_plot <- data.frame(
      id    = c(
        rep(task_names[1], n_plot),
        rep(task_names[2], n_plot),
        rep(task_names[3], n_plot)
      ),
      iff   = rep(iff_plot, 3),
      judge = rep(Judge, 3),
      cos   = as.vector(cosine_sim[, c(1, 2, 3)])
    )

We choose a color palette and fit `binsreg` with a grouping factor:

    pal <- c(
      "Coding Task"         = "#0072B2",
      "Concept Explanation" = "#D55E00",
      "Text Generation"     = "#009E73"
    )

    p <- binsreg::binsreg(
      y           = iff,
      x           = cos,
      by          = id,
      bycolors    = pal,
      bysymbols   = rep(16, 3),
      legendTitle = "Conversation task category",
      data        = dataset_plot,
      dots        = c(1, 1),
      cb          = c(2, 2),
      line        = c(1, 1),
      nsims       = 2000,
      simsgrid    = 100,
      randcut     = 1,
      cluster     = judge,
      masspoints  = "off"
    )$bins_plot

We clean up the plot and annotate it with example prompts.

    p <- p +
      scale_color_manual(values = pal) +
      scale_fill_manual(values = pal)

    ribbon_idx <- sapply(p$layers, function(ly) inherits(ly$geom, "GeomRibbon"))
    for (i in which(ribbon_idx)) {
      p$layers[[i]]$aes_params$alpha <- 0.75
    }

    notes_text <- paste0(
      "**Coding Task**: \"Write a Python moving average function.\"<br>",
      "**Concept Explanation**: \"Explain how interest rates affect inflation.\"<br>",
      "**Text Generation**: \"Rewrite this paragraph more professionally.\""
    )

    p_clean <- p +
      geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.375) +
      geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.375) +
      scale_y_continuous(n.breaks = 10, limits = c(-0.25, 0.75)) +
      scale_x_continuous(n.breaks = 10, limits = c(-0.75, 1)) +
      xlab("Cosine similarity") +
      ylab("Best model probability estimate and 95 percent UCB") +
      ggtitle(
        "Conditional Bradley Terry - Mistral Medium 3.1",
        subtitle = "Prompt specific heterogeneity by task similarity"
      ) +
      annotate("text", x = -0.5, y = Inf, label = "Less similar",
               hjust = 0.5, vjust = 1.2, size = 4, fontface = "italic") +
      annotate("text", x =  0.5, y = Inf, label = "More similar",
               hjust = 0.5, vjust = 1.2, size = 4, fontface = "italic") +
      labs(caption = notes_text) +
      theme_bw() +
      theme(
        plot.caption = ggtext::element_markdown(
          size   = 8,
          hjust  = 0,
          margin = margin(t = 8)
        )
      )

    p_clean

This graph shows how the best model probability varies as prompts become more similar to canonical coding, explanation, or text generation tasks.

---

## 12. Policy Trees for Model Selection

The influence functions can be interpreted as gradients of welfare functionals with respect to model assignment. This is exactly the type of object used in policy learning. We can feed them into `policytree` to learn interpretable deployment policies.

### 12.1 Policy for Cost Optimized Subject to Best Probability

Consider the influence matrix for cost times probability best, centered to remove constants:

    centered_iff_cost_best <- apply(iff_cost_best, 2, function(c) c - mean(c))

We learn a depth 1 policy tree that chooses the model with minimal cost times best:

    opt_bestcost <- policy_tree(X, -centered_iff_cost_best, depth = 1)
    newD_bestcost <- predict(opt_bestcost, X)

We then extract influence scores for the chosen model and summarize them via fixed effects regression clustered by judge:

    policy_iff_cost_bestcost <- iff_cost[cbind(1:nrow(iff_cost), newD_bestcost)]
    policy_iff_best_bestcost <- iff_best[cbind(1:nrow(iff_best), newD_bestcost)]

    results_bestcost <- tibble(
      cost  = policy_iff_cost_bestcost,
      best  = policy_iff_best_bestcost,
      judge = Judge
    )

    fixest::feols(c(cost, best) ~ 1, results_bestcost, cluster = ~judge)

The coefficients of this regression estimate the welfare under the learned policy.

### 12.2 Policy for Maximizing Best Probability

We can similarly focus on policies that maximize the probability of being best, ignoring energy for now:

    centered_iff_best <- apply(iff_best, 2, function(c) c - mean(c))

    opt_best   <- policy_tree(X, centered_iff_best, depth = 1)
    newD_best  <- predict(opt_best, X)

    policy_iff_cost_best <- iff_cost[cbind(1:nrow(iff_cost), newD_best)]
    policy_iff_best_best <- iff_best[cbind(1:nrow(iff_best), newD_best)]

    results_best <- tibble(
      cost  = policy_iff_cost_best,
      best  = policy_iff_best_best,
      judge = Judge
    )

    fixest::feols(c(cost, best) ~ 1, results_best, cluster = ~judge)

### 12.3 Comparing Cost Focused and Best Focused Policies

The difference in influence between the cost focused policy and the best focused policy can be summarized as:

    bestcost_vs_best <- tibble(
      cost  = policy_iff_cost_bestcost - policy_iff_cost_best,
      best  = policy_iff_best_bestcost - policy_iff_best_best,
      judge = Judge
    )

    fixest::feols(c(cost, best) ~ 1, bestcost_vs_best, cluster = ~judge)

The two coefficients correspond to the average difference in energy and best probability, respectively, when switching from a best probability policy to a cost adjusted policy. Confidence intervals make it straightforward to quantify tradeoffs.

---

## 13. Comparison With Classical Bradley Terry

To highlight the value of the semi parametric influence based approach, we compare it to a classical Bradley Terry regression that imposes constant coefficients across prompts.

### 13.1 Classical Bradley Terry With `fixest`

We use the design `D_pref` as in the neural net training and fit a simple logistic regression:

    y1 <- IF_pref
    colnames(y1) <- colnames(D_pref)

    se_pref <- apply(y1, 2, sd) / sqrt(nrow(y1))
    mu_pref <- apply(y1, 2, mean)

    umod <- fixest::feols(
      Y ~ D_pref - 1,
      tibble(Y = Y, D_pref = D_pref),
      vcov = "HC1"
    )

    se_um <- fixest::se(umod)
    mu_um <- coef(umod)

We assemble results into a single tibble with classical and heterogeneity enriched estimates:

    results <- tibble(
      Model   = model_names_pref_short,
      Mu_bt   = mu_um,
      CIl_bt  = mu_um - qnorm(0.975) * se_um,
      CIU_bt  = mu_um + qnorm(0.975) * se_um,
      Mu_het  = mu_pref,
      CIl_het = mu_pref - qnorm(0.975) * se_pref,
      CIU_het = mu_pref + qnorm(0.975) * se_pref
    ) %>%
      arrange(Mu_bt)

We reshape for plotting:

    results_long <- results %>%
      pivot_longer(
        cols        = -Model,
        names_to    = c(".value", "Type"),
        names_pattern = "^(Mu|CIl|CIU)_(bt|het)$"
      ) %>%
      mutate(
        Type = recode(Type,
                      bt  = "No heterogeneity",
                      het = "Heterogeneity enriched")
      )

    results_long <- results_long %>%
      mutate(Model = factor(Model, levels = results$Model))

    ggplot(results_long, aes(y = Model, x = Mu, color = Type)) +
      geom_point(position = position_dodge(width = 0.6), size = 2) +
      geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.25) +
      geom_errorbarh(
        aes(xmin = CIl, xmax = CIU),
        position = position_dodge(width = 0.6),
        height = 0.2
      ) +
      scale_color_manual(
        values = c(
          "No heterogeneity"       = "black",
          "Heterogeneity enriched" = "#0072B2"
        )
      ) +
      labs(
        y     = "Model name",
        x     = "Average preference parameter estimate and 95 percent CI",
        color = "Estimate type"
      ) +
      theme_bw() +
      theme(
        axis.text.y     = element_text(size = 8),
        legend.position = "right"
      ) +
      scale_x_continuous(n.breaks = 10, limits = c(-3, 3)) +
      ggtitle("Bradley Terry model results")

This plot overlays constant coefficient Bradley Terry estimates with semi parametric influence based averages, showing how accounting for heterogeneity in `X` can change the ranking and statistical significance of models.

---

## 14. Summary

This tutorial has constructed an end to end semi parametric Bradley Terry pipeline that:

1. Uses neural networks to model conditional preference and cost functions as functions of rich covariates.
2. Leverages the structure of the Bradley Terry design to build conditional Hessians that are symmetric positive definite and interpretable.
3. Employs cross fitting across judges to debias regularized neural nets in subsequent influence based inference.
4. Delivers influence functions for four functionals of interest:
   - Conditional preference coefficients,
   - Conditional probabilities of being best,
   - Conditional cost coefficients,
   - Conditional cost times probability best and related ratios.
5. Provides pointwise and uniform confidence bands across all models.
6. Enables heterogeneity analysis in covariates and prompt embedding space using standard nonparametric tools.
7. Translates influence function estimates into interpretable deployment rules via policy trees and evaluates the welfare of these rules.

The same blueprint can be extended to additional metrics (such as latency or safety scores), alternative link functions, or multi arm comparisons beyond pairwise settings, while preserving the central role of structural deep learning and influence function based inference.

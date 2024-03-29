library(R6)

LinseedMetadata <- R6Class(
  "LinseedMetadata",
  public = list(
      filtered_dataset = NULL,
      V_row = NULL,
      V_column = NULL,
      raw_dataset = NULL,
      path_ = NULL,
      analysis_name = NULL,
      dataset = NULL,
      top_genes = NULL,
      cell_types = NULL,
      coef_der_X = NULL,
      coef_der_Omega = NULL,
      coef_hinge_H = NULL,
      coef_hinge_W = NULL,
      coef_pos_D_h = NULL,
      coef_pos_D_w = NULL,
      global_iterations = NULL,

      init_count_neg_props = NULL,
      init_count_neg_basis = NULL,
      count_neg_props = NULL,
      count_neg_basis = NULL,

      full_proportions = NULL,
      full_basis = NULL,
      orig_full_proportions = NULL,
      orig_full_basis = NULL,
      R = NULL,
      S = NULL,

      distance_genes = NULL,
      distance_samples = NULL,
      mean_radius_X = NULL,
      mean_radius_Omega = NULL,
      
      final_X = NULL,
      final_Omega = NULL,
      final_W = NULL,
      final_H = NULL,
      final_D_h = NULL,
      final_D_w = NULL,
      
      init_D_w = NULL,
      init_D_h = NULL,
      init_X = NULL,
      init_H = NULL,
      init_W = NULL,
      init_Omega = NULL,

      errors_statistics = NULL,
      points_statistics_X = NULL,
      points_statistics_Omega = NULL,
      blocks_statistics = NULL,
      filters_pipeline = NULL,

      initialize = function(linseed_object) {
        self$filtered_dataset <- linseed_object$filtered_dataset
        self$raw_dataset <- linseed_object$raw_dataset
        self$V_row <- linseed_object$V_row
        self$V_column <- linseed_object$V_column

        self$dataset <- linseed_object$dataset
        self$path_ <- linseed_object$path_
        self$analysis_name <- linseed_object$analysis_name
        self$top_genes <- linseed_object$top_genes
        self$cell_types <- linseed_object$cell_types
        self$coef_der_X <- linseed_object$coef_der_X
        self$coef_der_Omega <- linseed_object$coef_der_Omega
        self$coef_hinge_H <- linseed_object$coef_hinge_H
        self$coef_hinge_W <- linseed_object$coef_hinge_W
        self$coef_pos_D_h <- linseed_object$coef_pos_D_h
        self$coef_pos_D_w <- linseed_object$coef_pos_D_w
        self$global_iterations <- linseed_object$global_iterations

        self$init_X <- linseed_object$init_X
        self$init_Omega <- linseed_object$init_Omega
        self$init_H <- linseed_object$init_H
        self$init_W <- linseed_object$init_W

        self$final_X <- linseed_object$X
        self$final_Omega <- linseed_object$Omega
        self$final_H <- linseed_object$H_
        self$final_W <- linseed_object$W_
        
        self$errors_statistics <- linseed_object$errors_statistics
        self$points_statistics_X <- linseed_object$points_statistics_X
        self$points_statistics_Omega <- linseed_object$points_statistics_Omega
        self$blocks_statistics <- linseed_object$blocks_statistics

        self$R <- linseed_object$R
        self$S <- linseed_object$S

        self$init_D_h <- linseed_object$init_D_h
        self$init_D_w <- linseed_object$init_D_w
        self$final_D_h <- linseed_object$D_h
        self$final_D_w <- linseed_object$D_w

        self$init_count_neg_props <- linseed_object$init_count_neg_props
        self$init_count_neg_basis <- linseed_object$init_count_neg_basis
        self$count_neg_props <- linseed_object$count_neg_props
        self$count_neg_basis <- linseed_object$count_neg_basis

        self$full_proportions <- linseed_object$full_proportions
        self$orig_full_proportions <- linseed_object$orig_full_proportions
        self$full_basis <- linseed_object$full_basis
        self$orig_full_basis <- linseed_object$orig_full_basis

        self$distance_genes <- linseed_object$distance_genes
        self$distance_samples <- linseed_object$distance_samples
        self$mean_radius_X <- linseed_object$mean_radius_X
        self$mean_radius_Omega <- linseed_object$mean_radius_Omega

        self$filters_pipeline <- linseed_object$filters_pipeline

      }
  )
  )

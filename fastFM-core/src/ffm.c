// Author: Immanuel Bayer
// License: BSD 3 clause

#include "fast_fm.h"
/*
 * Interface for external call
 * used by the python wrapper and cli interface
 */
const int MAX = 10010;
const int MAX_ITER = 1000;

void ffm_predict(double *w_0, double *w, double *V, cs *X, double *y_pred,
                 int k) {
  int n_samples = X->m;
  int n_features = X->n;
  ffm_vector ffm_w = {.size = n_features, .data = w, .owner = 0};
  ffm_matrix ffm_V = {.size0 = k, .size1 = n_features, .data = V, .owner = 0};
  ffm_coef coef = {.w_0 = *w_0, .w = &ffm_w, .V = &ffm_V};

  ffm_vector ffm_y_pred = {.size = n_samples, .data = y_pred, .owner = 0};
  sparse_predict(&coef, X, &ffm_y_pred);
}

void ffm_als_fit(double *w_0, double *w, double *V, cs *X, double *y,
                 ffm_param *param) {
  param->SOLVER = SOLVER_ALS;
  int n_samples = X->m;
  int n_features = X->n;

  ffm_vector ffm_w = {.size = n_features, .data = w, .owner = 0};
  ffm_matrix ffm_V = {
      .size0 = param->k, .size1 = n_features, .data = V, .owner = 0};
  ffm_coef coef = {
      .w_0 = *w_0, .w = &ffm_w, .V = &ffm_V, .lambda_w = param->init_lambda_w};
  if (param->k > 0) {
    coef.lambda_V = ffm_vector_alloc(param->k);
    coef.mu_V = ffm_vector_alloc(param->k);
    ffm_vector_set_all(coef.lambda_V, param->init_lambda_V);
  } else {
    coef.lambda_V = NULL;
    coef.mu_V = NULL;
  }

  ffm_vector ffm_y = {.size = n_samples, .data = y, .owner = 0};
  sparse_fit(&coef, X, NULL, &ffm_y, NULL, *param);

  // copy the last coef values back into the python memory
  *w_0 = coef.w_0;

  if (param->k > 0) ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}

void ffm_mcmc_fit_predict(double *w_0, double *w, double *V, cs *X_train,
                          cs *X_test, double *y_train, double *y_pred,
                          ffm_param *param) {
  param->SOLVER = SOLVER_MCMC;
  int k = param->k;
  double *hyper_param = param->hyper_param;
  int n_test_samples = X_test->m;
  int n_train_samples = X_train->m;
  int n_features = X_train->n;
  ffm_vector ffm_w = {.size = n_features, .data = w, .owner = 0};
  ffm_matrix ffm_V = {
      .size0 = param->k, .size1 = n_features, .data = V, .owner = 0};
  ffm_coef coef = {.w_0 = *w_0,
                   .w = &ffm_w,
                   .V = &ffm_V,
                   .lambda_w = param->init_lambda_w,
                   .alpha = 1,
                   .mu_w = 0};
  if (k > 0) {
    coef.lambda_V = ffm_vector_alloc(param->k);
    coef.mu_V = ffm_vector_alloc(param->k);
  } else {
    coef.lambda_V = NULL;
    coef.mu_V = NULL;
  }

  // set inital values for hyperparameter
  int w_groups = 1;
  assert(param->n_hyper_param == 1 + 2 * k + 2 * w_groups &&
         "hyper_parameter vector has wrong size");
  if (param->warm_start) {
    coef.alpha = hyper_param[0];
    coef.lambda_w = hyper_param[1];
    // copy V lambda's over
    for (int i = 0; i < k; i++)
      ffm_vector_set(coef.lambda_V, i, hyper_param[i + 1 + w_groups]);
    coef.mu_w = hyper_param[k + 1 + w_groups];
    // copy V mu's over
    for (int i = 0; i < k; i++)
      ffm_vector_set(coef.mu_V, i, hyper_param[i + 1 + (2 * w_groups) + k]);
  }

  ffm_vector ffm_y_train = {
      .size = n_train_samples, .data = y_train, .owner = 0};
  ffm_vector ffm_y_pred = {.size = n_test_samples, .data = y_pred, .owner = 0};
  sparse_fit(&coef, X_train, X_test, &ffm_y_train, &ffm_y_pred, *param);
  // copy the last coef values back into the python memory
  *w_0 = coef.w_0;

  // copy current hyperparameter back
  hyper_param[0] = coef.alpha;
  hyper_param[1] = coef.lambda_w;
  // copy V lambda's back
  for (int i = 0; i < k; i++)
    hyper_param[i + 1 + w_groups] = ffm_vector_get(coef.lambda_V, i);
  hyper_param[k + 1 + w_groups] = coef.mu_w;
  // copy mu's back
  for (int i = 0; i < k; i++)
    hyper_param[i + 1 + (2 * w_groups) + k] = ffm_vector_get(coef.mu_V, i);

  if (k > 0) ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}

void ffm_sgd_bpr_fit(double *w_0, double *w, double *V, cs *X, double *pairs,
                     int n_pairs, ffm_param *param) {
  // X is transpose of design matrix. Samples are stored in columns.
  int n_features = X->m;
  ffm_vector ffm_w = {.size = n_features, .data = w, .owner = 0};
  ffm_matrix ffm_V = {
      .size0 = param->k, .size1 = n_features, .data = V, .owner = 0};
  ffm_coef coef = {
      .w_0 = *w_0, .w = &ffm_w, .V = &ffm_V, .lambda_w = param->init_lambda_w};
  if (param->k > 0) {
    coef.lambda_V = ffm_vector_alloc(param->k);
    coef.mu_V = ffm_vector_alloc(param->k);
  } else {
    coef.lambda_V = NULL;
    coef.mu_V = NULL;
  }

  ffm_matrix ffm_y = {.size0 = n_pairs, .size1 = 2, .data = pairs, .owner = 0};
  ffm_fit_sgd_bpr(&coef, X, &ffm_y, *param);

  // copy the last coef values back into the python memory
  *w_0 = coef.w_0;
  if (param->k > 0) ffm_vector_free_all(coef.lambda_V, coef.mu_V);
}

void ffm_sgd_fit(double *w_0, double *w, double *V, cs *X, double *y,

                 ffm_param *param) {

  // X is transpose of design matrix. Samples are stored in columns.

  int n_samples = X->n;

  int n_features = X->m;

  int k;
  k = param->k;

  ffm_vector ffm_w = {.size = n_features, .data = w, .owner = 0};

  ffm_matrix ffm_V = {

      .size0 = param->k, .size1 = n_features, .data = V, .owner = 0};

  ffm_coef coef = {

      .w_0 = *w_0, .w = &ffm_w, .V = &ffm_V, .lambda_w = param->init_lambda_w};

  if (param->k > 0) {

    coef.lambda_V = ffm_vector_alloc(param->k);

    coef.mu_V = ffm_vector_alloc(param->k);

  } else {

    coef.lambda_V = NULL;

    coef.mu_V = NULL;

  }

  ffm_vector ffm_y = {.size = n_samples, .data = y, .owner = 0};

  ffm_fit_sgd(&coef, X, &ffm_y, param);
  
  for(int f = 0; f < n_features; f++){
      double t = ffm_vector_get( coef.w, f);
      if (t > 0.000000)       
        ffm_vector_set( coef.w, f, 1);     
      else    
        ffm_vector_set( coef.w, f, -1);    
  }

  for(int j = 0; j < k; j++){
    for(int f = 0; f < n_features; f++){
        double t = ffm_matrix_get( coef.V, j, f);
        if (t > 0.000000)           
            ffm_matrix_set( coef.V, j, f, 1);      
        else
            ffm_matrix_set( coef.V, j, f, -1);       
    }
  } 

  ffm_vector *y_pred = ffm_vector_calloc(n_samples);
  ffm_vector *y_pred_t = ffm_vector_calloc(n_samples);
  //col_predict(&coef, X, y_pred);
  //for(int i = 0; i < n_samples; i++)
   // printf("%f %f\n",y[i],y_pred->data[i]);
  //printf("\n");
  int p, j, f, n, *Ap, *Ai;
  double *Ax;
  n = X->n;
  Ap = X->p;
  Ai = X->i;
  Ax = X->x;
  int c,ch;
  c = 1;
  while(c)
  {
    c = 0;
    
    ffm_vector_set_all(y_pred_t, coef.w_0);
    for (j = 0; j < n; j++)
      for (p = Ap[j]; p < Ap[j + 1]; p++) {
        double tmp_x = Ax[p];
        y_pred_t->data[j] -= 0.5 * (tmp_x * tmp_x) * k;
      }
    V_predict(&coef, X, y_pred_t);
    ch = 1;
    while (ch){  
      ch = 0;
      //for (j = 0; j < X->n; j++)
        //y_pred_t->data[j] = 0;
      /*ffm_vector_set_all(y_pred_t, coef.w_0);
      for (j = 0; j < n; j++)
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
          double tmp_x = Ax[p];
          y_pred_t->data[j] -= 0.5 * (tmp_x * tmp_x) * k;
        }
      V_predict(&coef, X, y_pred_t);*/
      for(int f = 0; f < n_features; f++){
  
          //col_predict(&coef, X, y_pred);
          //y_pred = y_pred_t;
          for (j = 0; j < n; j++)
            y_pred->data[j] = y_pred_t->data[j];
          Cs_row_gaxpy(X, coef.w->data, y_pred->data);
            
          double loss1 = 0.0;
          for(int i = 0; i < n_samples; i++)
              if ((1 - y[i] * y_pred->data[i]) > 0)
                  loss1 += 1 - y[i] * y_pred->data[i];
  
          double t = ffm_vector_get( coef.w, f);
          ffm_vector_set( coef.w, f, -t);
          
          //col_predict(&coef, X, y_pred);
          for (j = 0; j < n; j++)
            y_pred->data[j] = y_pred_t->data[j];
          Cs_row_gaxpy(X, coef.w->data, y_pred->data);
            
          double loss2 = 0.0;
          for(int i = 0; i < n_samples; i++)
              if ((1 - y[i] * y_pred->data[i]) > 0)
                  loss2 += 1 - y[i] * y_pred->data[i];
  
          if (loss2 >= loss1){
              double t = ffm_vector_get( coef.w, f);
              ffm_vector_set( coef.w, f, -t);
          }
          else ch++,c++;
      }
    }
    printf("c1:%d\n", c);
    
    ffm_vector_set_all(y_pred_t, coef.w_0);
    Cs_row_gaxpy(X, coef.w->data, y_pred_t->data);
    for (j = 0; j < n; j++)
      for (p = Ap[j]; p < Ap[j + 1]; p++) {
        double tmp_x = Ax[p];
        y_pred_t->data[j] -= 0.5 * (tmp_x * tmp_x) * k;
      }
    ch = 1;
    while(ch){
      ch = 0;
     /* ffm_vector_set_all(y_pred_t, coef.w_0);
      Cs_row_gaxpy(X, coef.w->data, y_pred_t->data);
      for (j = 0; j < n; j++)
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
          double tmp_x = Ax[p];
          y_pred_t->data[j] -= 0.5 * (tmp_x * tmp_x) * k;
        }*/
      for(int j = 0; j < k; j++){
          for(int f = 0; f < n_features; f++){
          
              //col_predict(&coef, X, y_pred);
              for (int i = 0; i < n; i++)
                  y_pred->data[i] = y_pred_t->data[i];
              V_predict(&coef, X, y_pred);
              
              double loss1 = 0.0;
              for(int i =0; i < n_samples; i++)
                  if ((1 - y[i] * y_pred->data[i]) > 0)
                      loss1 += 1 - y[i] * y_pred->data[i];  
                         
              double t = ffm_matrix_get( coef.V, j, f);
              ffm_matrix_set( coef.V, j, f, -t);  
              //col_predict(&coef, X, y_pred); 
              for (int i = 0; i < n; i++)
                  y_pred->data[i] = y_pred_t->data[i];
              V_predict(&coef, X, y_pred);
                    
              double loss2 = 0.0;        
              for(int i =0; i < n_samples; i++)
                  if ((1 - y[i] * y_pred->data[i]) > 0)
                      loss2 += 1 - y[i] * y_pred->data[i];    
              if (loss2 >= loss1){  
                  double t = ffm_matrix_get( coef.V, j, f);
                  ffm_matrix_set( coef.V, j, f, -t);          
              }
              else ch++,c++;          
          }
      }
    }
    printf("c2:%d\n", c);
  }
  /*printf("============1=============\n");
  for(int f = 0; f < n_features; f++){

        
      printf("%f\n", w[f]);
  }
  printf("============2==============\n");
    for(int j = 0; j < k; j++){
    for(int f = 0; f < n_features; f++){
       
       printf("%f\n", V[j * n_features + f]);
    }
    printf("\n");
  }*/

  *w_0 = coef.w_0;

  if (param->k > 0) ffm_vector_free_all(coef.lambda_V, coef.mu_V);

}

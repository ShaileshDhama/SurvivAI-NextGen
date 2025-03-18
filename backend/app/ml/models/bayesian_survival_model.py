"""
Bayesian Survival Models implementation
Probabilistic survival time estimation using PyMC
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import pymc as pm
import arviz as az
from scipy import stats

from app.ml.models.base_model import BaseSurvivalModel
from app.models.analysis import SurvivalCurve, FeatureImportance


class BayesianSurvivalModel(BaseSurvivalModel):
    """
    Bayesian Survival Models implementation using PyMC
    Probabilistic survival time estimation with uncertainty quantification
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize with model parameters"""
        super().__init__(model_params)
        self.model_params = model_params or {}
        self.model = None
        self.trace = None
        self.feature_names = None
        self.baseline_hazard = None
        self.distribution = self.model_params.get('distribution', 'weibull')
        self.num_samples = self.model_params.get('num_samples', 1000)
        self.num_chains = self.model_params.get('num_chains', 2)
        self.tune = self.model_params.get('tune', 1000)
        self.times = None
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, E: np.ndarray) -> "BayesianSurvivalModel":
        """
        Fit Bayesian survival model to the data
        
        Args:
            X: Covariates DataFrame
            T: Time to event
            E: Event indicator (1 if event occurred, 0 if censored)
            
        Returns:
            self: Fitted model
        """
        self.feature_names = X.columns.tolist()
        self.times = T
        
        # Create design matrix
        X_design = X.values
        
        # Define model based on chosen distribution
        with pm.Model() as self.model:
            # Create priors for coefficients
            # Use weakly informative priors
            beta = pm.Normal('beta', 
                          mu=0, 
                          sigma=self.model_params.get('prior_sigma', 2.0),
                          shape=X_design.shape[1])
            
            # Linear predictor
            eta = pm.math.dot(X_design, beta)
            
            # Define survival distribution
            if self.distribution == 'weibull':
                # Weibull distribution parameters
                alpha = pm.HalfNormal('alpha', sigma=2.0)  # shape parameter
                lambda_0 = pm.HalfNormal('lambda_0', sigma=2.0)  # scale parameter
                
                # Adjust rate based on covariates
                lambda_i = lambda_0 * pm.math.exp(eta)
                
                # Define likelihood for non-censored data
                obs_surv = pm.Weibull('obs_surv', 
                                     alpha=alpha, 
                                     beta=lambda_i[E.astype(bool)], 
                                     observed=T[E.astype(bool)])
                
                # Define likelihood for censored data (if any)
                if np.any(~E.astype(bool)):
                    cens_surv = pm.Potential('cens_surv', 
                                           pm.math.log(pm.math.exp(
                                               -pm.math.pow(T[~E.astype(bool)] * lambda_i[~E.astype(bool)], alpha)
                                           )))
            
            elif self.distribution == 'exponential':
                # Exponential distribution parameter
                lambda_0 = pm.HalfNormal('lambda_0', sigma=2.0)  # scale parameter
                
                # Adjust rate based on covariates
                lambda_i = lambda_0 * pm.math.exp(eta)
                
                # Define likelihood for non-censored data
                obs_surv = pm.Exponential('obs_surv', 
                                        lam=lambda_i[E.astype(bool)], 
                                        observed=T[E.astype(bool)])
                
                # Define likelihood for censored data (if any)
                if np.any(~E.astype(bool)):
                    cens_surv = pm.Potential('cens_surv', 
                                           pm.math.log(pm.math.exp(
                                               -lambda_i[~E.astype(bool)] * T[~E.astype(bool)]
                                           )))
            
            elif self.distribution == 'lognormal':
                # Lognormal distribution parameters
                sigma = pm.HalfNormal('sigma', sigma=1.0)
                
                # Define likelihood for non-censored data
                obs_surv = pm.LogNormal('obs_surv', 
                                      mu=eta[E.astype(bool)], 
                                      sigma=sigma, 
                                      observed=T[E.astype(bool)])
                
                # Define likelihood for censored data (if any)
                if np.any(~E.astype(bool)):
                    cens_surv = pm.Potential('cens_surv', 
                                           pm.math.log(1 - pm.math.exp(
                                               pm.math.log_normal_cdf(T[~E.astype(bool)], 
                                                                  mu=eta[~E.astype(bool)], 
                                                                  sigma=sigma)
                                           )))
            
            # Sample from posterior
            self.trace = pm.sample(
                draws=self.num_samples,
                chains=self.num_chains,
                tune=self.tune,
                cores=self.model_params.get('cores', 1),
                return_inferencedata=True
            )
        
        # Estimate baseline hazard for prediction
        self._estimate_baseline_hazard(T, E)
        
        self.fitted = True
        return self
    
    def _estimate_baseline_hazard(self, T, E):
        """Estimate baseline hazard function"""
        # Use Nelson-Aalen estimator for baseline hazard
        sorted_times = np.sort(np.unique(T[E.astype(bool)]))
        self.baseline_times = sorted_times
        
        # Get posterior samples of coefficients
        beta_samples = self.trace.posterior['beta'].values
        mean_beta = beta_samples.mean(axis=(0, 1))
        
        # Calculate baseline hazard using Breslow's method
        if self.distribution == 'weibull':
            alpha_samples = self.trace.posterior['alpha'].values
            lambda_0_samples = self.trace.posterior['lambda_0'].values
            mean_alpha = alpha_samples.mean()
            mean_lambda_0 = lambda_0_samples.mean()
            
            # For Weibull, baseline hazard is: h0(t) = alpha * lambda_0 * t^(alpha-1)
            self.baseline_hazard = mean_alpha * mean_lambda_0 * np.power(sorted_times, mean_alpha - 1)
            self.baseline_cumhazard = mean_lambda_0 * np.power(sorted_times, mean_alpha)
        
        elif self.distribution == 'exponential':
            lambda_0_samples = self.trace.posterior['lambda_0'].values
            mean_lambda_0 = lambda_0_samples.mean()
            
            # For exponential, baseline hazard is constant: h0(t) = lambda_0
            self.baseline_hazard = np.ones_like(sorted_times) * mean_lambda_0
            self.baseline_cumhazard = mean_lambda_0 * sorted_times
        
        elif self.distribution == 'lognormal':
            # For lognormal, use numerical approximation
            sigma_samples = self.trace.posterior['sigma'].values
            mean_sigma = sigma_samples.mean()
            
            # Calculate hazard: h(t) = f(t) / S(t)
            pdf = stats.lognorm.pdf(sorted_times, s=mean_sigma, scale=np.exp(0))
            sf = stats.lognorm.sf(sorted_times, s=mean_sigma, scale=np.exp(0))
            self.baseline_hazard = pdf / sf
            
            # Calculate cumulative hazard: H(t) = -log(S(t))
            self.baseline_cumhazard = -np.log(sf)
    
    def predict_survival_function(self, X: pd.DataFrame) -> List[SurvivalCurve]:
        """
        Predict survival function for given covariates
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            List of survival curves
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get posterior samples of parameters
        beta_samples = self.trace.posterior['beta'].values.reshape((-1, len(self.feature_names)))
        
        # Create prediction times
        if self.model_params.get('pred_time_max'):
            pred_times = np.linspace(0, self.model_params.get('pred_time_max'), 100)
        else:
            pred_times = np.linspace(0, np.max(self.times) * 1.2, 100)
        
        survival_curves = []
        
        # For each subject, compute survival function
        for i, row in X.iterrows():
            # Get linear predictor for this subject
            eta = np.dot(row.values, beta_samples.mean(axis=0))
            
            # Calculate survival function based on distribution
            if self.distribution == 'weibull':
                alpha_samples = self.trace.posterior['alpha'].values.flatten()
                lambda_0_samples = self.trace.posterior['lambda_0'].values.flatten()
                
                mean_alpha = alpha_samples.mean()
                mean_lambda_0 = lambda_0_samples.mean()
                
                # S(t) = exp(-(lambda_0 * t)^alpha * exp(eta))
                lambda_i = mean_lambda_0 * np.exp(eta)
                surv_prob = np.exp(-np.power(pred_times * lambda_i, mean_alpha))
                
                # Calculate 95% credible intervals
                surv_samples = np.zeros((len(beta_samples), len(pred_times)))
                for j, (beta, alpha, lambda_0) in enumerate(zip(beta_samples, 
                                                              alpha_samples, 
                                                              lambda_0_samples)):
                    eta_j = np.dot(row.values, beta)
                    lambda_ij = lambda_0 * np.exp(eta_j)
                    surv_samples[j] = np.exp(-np.power(pred_times * lambda_ij, alpha))
                
                # Get credible intervals
                lower = np.percentile(surv_samples, 2.5, axis=0)
                upper = np.percentile(surv_samples, 97.5, axis=0)
                
            elif self.distribution == 'exponential':
                lambda_0_samples = self.trace.posterior['lambda_0'].values.flatten()
                mean_lambda_0 = lambda_0_samples.mean()
                
                # S(t) = exp(-lambda_0 * t * exp(eta))
                lambda_i = mean_lambda_0 * np.exp(eta)
                surv_prob = np.exp(-lambda_i * pred_times)
                
                # Calculate 95% credible intervals
                surv_samples = np.zeros((len(beta_samples), len(pred_times)))
                for j, (beta, lambda_0) in enumerate(zip(beta_samples, lambda_0_samples)):
                    eta_j = np.dot(row.values, beta)
                    lambda_ij = lambda_0 * np.exp(eta_j)
                    surv_samples[j] = np.exp(-lambda_ij * pred_times)
                
                # Get credible intervals
                lower = np.percentile(surv_samples, 2.5, axis=0)
                upper = np.percentile(surv_samples, 97.5, axis=0)
                
            elif self.distribution == 'lognormal':
                sigma_samples = self.trace.posterior['sigma'].values.flatten()
                mean_sigma = sigma_samples.mean()
                
                # S(t) = 1 - CDF(t)
                surv_prob = 1 - stats.lognorm.cdf(pred_times, s=mean_sigma, scale=np.exp(eta))
                
                # Calculate 95% credible intervals
                surv_samples = np.zeros((len(beta_samples), len(pred_times)))
                for j, (beta, sigma) in enumerate(zip(beta_samples, sigma_samples)):
                    eta_j = np.dot(row.values, beta)
                    surv_samples[j] = 1 - stats.lognorm.cdf(pred_times, s=sigma, scale=np.exp(eta_j))
                
                # Get credible intervals
                lower = np.percentile(surv_samples, 2.5, axis=0)
                upper = np.percentile(surv_samples, 97.5, axis=0)
            
            # Create survival curve object
            survival_curves.append(
                SurvivalCurve(
                    times=pred_times.tolist(),
                    survival_probs=surv_prob.tolist(),
                    confidence_intervals_lower=lower.tolist(),
                    confidence_intervals_upper=upper.tolist(),
                    group_name=f"Subject {i}"
                )
            )
        
        return survival_curves
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores for given covariates
        Higher values indicate higher risk (lower survival)
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of risk scores
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get posterior means of coefficients
        beta_samples = self.trace.posterior['beta'].values
        mean_beta = beta_samples.mean(axis=(0, 1))
        
        # Calculate risk scores (linear predictor)
        return np.exp(np.dot(X.values, mean_beta))
    
    def predict_median_survival_time(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median survival time for given covariates
        
        Args:
            X: Covariates DataFrame
            
        Returns:
            Array of median survival times
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get posterior means of parameters
        beta_samples = self.trace.posterior['beta'].values
        mean_beta = beta_samples.mean(axis=(0, 1))
        
        median_times = np.zeros(len(X))
        
        # Calculate median survival time based on distribution
        if self.distribution == 'weibull':
            alpha_samples = self.trace.posterior['alpha'].values
            lambda_0_samples = self.trace.posterior['lambda_0'].values
            mean_alpha = alpha_samples.mean()
            mean_lambda_0 = lambda_0_samples.mean()
            
            # For Weibull, median time is: t_median = (log(2) / (lambda_0 * exp(eta)))^(1/alpha)
            log_2 = np.log(2)
            
            for i, row in X.iterrows():
                eta = np.dot(row.values, mean_beta)
                lambda_i = mean_lambda_0 * np.exp(eta)
                median_times[i] = np.power(log_2 / lambda_i, 1 / mean_alpha)
        
        elif self.distribution == 'exponential':
            lambda_0_samples = self.trace.posterior['lambda_0'].values
            mean_lambda_0 = lambda_0_samples.mean()
            
            # For exponential, median time is: t_median = log(2) / (lambda_0 * exp(eta))
            log_2 = np.log(2)
            
            for i, row in X.iterrows():
                eta = np.dot(row.values, mean_beta)
                lambda_i = mean_lambda_0 * np.exp(eta)
                median_times[i] = log_2 / lambda_i
        
        elif self.distribution == 'lognormal':
            sigma_samples = self.trace.posterior['sigma'].values
            mean_sigma = sigma_samples.mean()
            
            # For lognormal, median time is: t_median = exp(eta)
            for i, row in X.iterrows():
                eta = np.dot(row.values, mean_beta)
                median_times[i] = np.exp(eta)
        
        return median_times
    
    def get_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from the fitted model
        
        Returns:
            Feature importance data
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Get posterior samples of coefficients
        beta_samples = self.trace.posterior['beta'].values.reshape((-1, len(self.feature_names)))
        
        # Calculate posterior means and standard deviations
        beta_means = beta_samples.mean(axis=0)
        beta_stds = beta_samples.std(axis=0)
        
        # Calculate 95% credible intervals
        beta_lower = np.percentile(beta_samples, 2.5, axis=0)
        beta_upper = np.percentile(beta_samples, 97.5, axis=0)
        
        # Calculate posterior probability that coefficient > 0
        prob_positive = np.mean(beta_samples > 0, axis=0)
        
        # Calculate hazard ratios (exp(beta))
        hr_samples = np.exp(beta_samples)
        hr_means = hr_samples.mean(axis=0)
        hr_lower = np.percentile(hr_samples, 2.5, axis=0)
        hr_upper = np.percentile(hr_samples, 97.5, axis=0)
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_values=beta_means.tolist(),
            importance_type="coefficient",
            additional_metrics={
                "coefficient_std": beta_stds.tolist(),
                "coefficient_lower_ci": beta_lower.tolist(),
                "coefficient_upper_ci": beta_upper.tolist(),
                "probability_positive": prob_positive.tolist(),
                "hazard_ratio": hr_means.tolist(),
                "hazard_ratio_lower_ci": hr_lower.tolist(),
                "hazard_ratio_upper_ci": hr_upper.tolist()
            }
        )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary of model summary statistics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        
        # Calculate posterior predictive checks
        if self.model_params.get('compute_ppc', True):
            try:
                with self.model:
                    ppc = pm.sample_posterior_predictive(self.trace)
                    waic = az.waic(self.trace, self.model)
                    loo = az.loo(self.trace, self.model)
                    
                    summary_stats = {
                        "waic": float(waic.waic),
                        "loo": float(loo.loo),
                        "p_waic": float(waic.p_waic),
                        "p_loo": float(loo.p_loo)
                    }
            except:
                summary_stats = {
                    "ppc_error": "Failed to compute posterior predictive checks"
                }
        else:
            summary_stats = {}
        
        # Get parameter estimates
        beta_samples = self.trace.posterior['beta'].values.reshape((-1, len(self.feature_names)))
        beta_means = beta_samples.mean(axis=0)
        beta_stds = beta_samples.std(axis=0)
        
        param_summary = {}
        for i, name in enumerate(self.feature_names):
            param_summary[name] = {
                "mean": float(beta_means[i]),
                "std": float(beta_stds[i]),
                "2.5%": float(np.percentile(beta_samples[:, i], 2.5)),
                "97.5%": float(np.percentile(beta_samples[:, i], 97.5))
            }
        
        # Get distribution parameters
        if self.distribution == 'weibull':
            alpha_samples = self.trace.posterior['alpha'].values.flatten()
            lambda_0_samples = self.trace.posterior['lambda_0'].values.flatten()
            
            dist_params = {
                "alpha": {
                    "mean": float(alpha_samples.mean()),
                    "std": float(alpha_samples.std()),
                    "2.5%": float(np.percentile(alpha_samples, 2.5)),
                    "97.5%": float(np.percentile(alpha_samples, 97.5))
                },
                "lambda_0": {
                    "mean": float(lambda_0_samples.mean()),
                    "std": float(lambda_0_samples.std()),
                    "2.5%": float(np.percentile(lambda_0_samples, 2.5)),
                    "97.5%": float(np.percentile(lambda_0_samples, 97.5))
                }
            }
        elif self.distribution == 'exponential':
            lambda_0_samples = self.trace.posterior['lambda_0'].values.flatten()
            
            dist_params = {
                "lambda_0": {
                    "mean": float(lambda_0_samples.mean()),
                    "std": float(lambda_0_samples.std()),
                    "2.5%": float(np.percentile(lambda_0_samples, 2.5)),
                    "97.5%": float(np.percentile(lambda_0_samples, 97.5))
                }
            }
        elif self.distribution == 'lognormal':
            sigma_samples = self.trace.posterior['sigma'].values.flatten()
            
            dist_params = {
                "sigma": {
                    "mean": float(sigma_samples.mean()),
                    "std": float(sigma_samples.std()),
                    "2.5%": float(np.percentile(sigma_samples, 2.5)),
                    "97.5%": float(np.percentile(sigma_samples, 97.5))
                }
            }
        
        # Complete summary
        summary = {
            "distribution": self.distribution,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "tune": self.tune,
            "parameters": param_summary,
            "distribution_parameters": dist_params,
            **summary_stats
        }
        
        return summary
    
    def save(self, path: str) -> str:
        """
        Save model to disk
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path to saved model
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save trace
        trace_path = os.path.join(path, f"bayesian_survival_{self.distribution}_trace.nc")
        self.trace.to_netcdf(trace_path)
        
        # Save other model data
        model_path = os.path.join(path, f"bayesian_survival_{self.distribution}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'distribution': self.distribution,
                'baseline_times': self.baseline_times,
                'baseline_hazard': self.baseline_hazard,
                'baseline_cumhazard': self.baseline_cumhazard,
                'fitted': self.fitted,
                'model_params': self.model_params,
                'trace_path': trace_path
            }, f)
        
        return model_path
    
    @classmethod
    def load(cls, path: str) -> "BayesianSurvivalModel":
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(model_params=data['model_params'])
        model.feature_names = data['feature_names']
        model.distribution = data['distribution']
        model.baseline_times = data['baseline_times']
        model.baseline_hazard = data['baseline_hazard']
        model.baseline_cumhazard = data['baseline_cumhazard']
        model.fitted = data['fitted']
        
        # Load trace
        model.trace = az.InferenceData.from_netcdf(data['trace_path'])
        
        return model

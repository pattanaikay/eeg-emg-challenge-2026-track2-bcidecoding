import scipy.special as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import os
from SincAdaptNet import SincAdaptNet
from tqdm import tqdm
import pandas as pd
import logging


class EEGModel(nn.Module):

    def __init__(
            self, feature_extractor: nn.Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, batched_inputs):
        x = self.feature_extractor.conv_spatial(batched_inputs)
        x1 = x.permute(0, 2, 1, 3)
        x2 = self.feature_extractor.conv_temporal(x1)

        x2 = x2.reshape(-1, 24 * 16, x1.shape[-1]).permute(0, 2, 1)
        x = self.feature_extractor.LN1(x2).permute(0, 2, 1).reshape(-1, 24, 16, x1.shape[-1])

        x = nn.functional.elu(self.feature_extractor.conv_class1(x))
        x = self.feature_extractor.LN2(x.permute(0, 2, 1, 3).reshape(-1, 32, x1.shape[-1]).permute(0, 2, 1))

        pseudo_labels = self.feature_extractor.softmax(
            self.feature_extractor.conv_class4(x.permute(0, 2, 1).reshape(-1, 1, 32, x1.shape[-1]))).squeeze(2)
        pseudo_labels_mean = torch.mean(pseudo_labels, -1)

        return pseudo_labels, pseudo_labels_mean


class DirichletEstimator:
    def __init__(self, tol=1e-4, max_iter=100000):
        """
        Dirichlet parameter estimator using fixed-point iteration.

        :param tol: Tolerance for convergence.
        :param max_iter: Maximum number of iterations.
        """
        self.tol = tol
        self.max_iter = max_iter

    def estimate(self, data, responsibilities=None):
        """
        Estimate the Dirichlet distribution parameters using fixed-point iteration.

        :param data: Observed data, shape (K, N), where N is the number of samples,
                     and K is the number of dimensions.
        :return: Estimated Dirichlet parameters, shape (K,).
        """
        K, N = data.shape
        if responsibilities is None:
            responsibilities = np.ones(N)
        log_p_bar = np.mean(responsibilities.squeeze() * np.log(data), axis=-1)

        # Initialize alpha with the method of moments
        alpha = self._initialize_alpha(data)

        for iteration in range(self.max_iter):
            alpha_sum = np.sum(alpha)

            # Fixed-point update
            alpha_new = self._inverse_digamma(
                sp.digamma(alpha_sum) + log_p_bar
            )

            # Check convergence
            if np.linalg.norm(alpha_new - alpha) < self.tol:
                return alpha_new

            alpha_old = alpha
            alpha = alpha_new

        raise RuntimeError("Fixed-point iteration did not converge within the maximum iterations")

    def _initialize_alpha(self, data):
        """
        Initialize alpha using the method of moments.

        :param data: Observed data, shape (K, N).
        :return: Initial alpha, shape (K,).
        """
        K, N = data.shape
        mean_p = np.mean(data, axis=-1)
        variance_p = np.var(data, axis=-1)
        mean_total = np.sum(mean_p)

        # alpha_0 = mean_total ** 2 / (np.sum(variance_p) - mean_total)
        alpha_0 = (mean_p * (1 - mean_p) / variance_p - mean_total).mean()
        return alpha_0 * mean_p

    def _inverse_digamma(self, y):
        """
        Approximate the inverse of the digamma function.

        :param y: Input values, shape (K,).
        :return: Approximations of the inverse digamma function, shape (K,).
        """
        # Initialization based on asymptotic expansion
        x = np.where(y >= -2.22, np.exp(y) + 0.5, -1 / (y + sp.psi(1)))

        # Newton iterations to refine the solution
        for _ in range(10):
            x = x - (sp.digamma(x) - y) / sp.polygamma(1, x)  # polygamma(1, x) is the trigamma function

        return x


from sklearn.mixture import GaussianMixture


class OnlineClustererGMM:
    def __init__(
            self,
            num_components=4,
            num_classes=2,
            tol=1e-4,
            max_iter=1000,
            buffer_size=1000,
            random_state=42,
            conf_threshold=0.52,
            entropy_threshold=0.69,
    ):
        self.num_components = num_components
        self.num_classes = num_classes
        self.tol = tol
        self.max_iter = max_iter
        self.buffer_size = buffer_size
        self.random_state = random_state

        self.conf_threshold = conf_threshold
        self.entropy_threshold = entropy_threshold

        # Initialize buffers and GMMs for each class
        self.buffers = {i: [] for i in range(num_classes)}
        self.gmm_models = {
            i: GaussianMixture(
                n_components=self.num_components,
                covariance_type='full',
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            for i in range(num_classes)
        }

        # To store fitted parameters
        self.cluster_centers = {i: None for i in range(num_classes)}
        self.mixture_weights = {i: None for i in range(num_classes)}
        self.cluster_cov = {i: None for i in range(num_classes)}

    def add_sample(self, alpha_estimate, pred, prob_map):
        confidence = prob_map[:, pred]
        entropy = -np.sum(prob_map * np.log(prob_map + 1e-10))

        if confidence >= self.conf_threshold and entropy <= self.entropy_threshold:
            if len(self.buffers[pred]) >= self.buffer_size:
                self.buffers[pred].pop(0)
            self.buffers[pred].append((alpha_estimate, pred, prob_map))
        return confidence, entropy

    def update(self, alpha_estimate, pred, prob_map):
        """
        Update the clusterer with a new sample's Dirichlet parameters.

        :param alpha_estimate: Estimated Dirichlet parameters for the sample, shape (K,).
        :param pred: Predicted class label for the sample, integer in [0, num_classes-1].
        """

        if len(self.buffers[pred]) >= 32:
            for pred in range(self.num_classes):
                samples = self.buffers[pred]

                buffer_all = np.array([sample[0] for sample in self.buffers[pred]])  # Shape: (buffer_size, K)

                if len(self.buffers[pred]) >= self.gmm_models[pred].n_components:
                    self.gmm_models[pred].fit(buffer_all)

                    # Store the fitted parameters
                    self.cluster_centers[pred] = self.gmm_models[pred].means_
                    self.mixture_weights[pred] = self.gmm_models[pred].weights_
                    self.cluster_cov[pred] = self.gmm_models[pred].covariances_

    def predict_class(self, alpha_estimate, prob_map):
        p_GMM_all = []
        for y_prime in range(self.num_classes):
            means_prime, weights_prime = self.get_cluster_parameters(y_prime)
            if means_prime is None or weights_prime is None:
                p_GMM_all.append(0)
                continue
            p_GMM_y_prime = 0
            for k in range(len(weights_prime)):
                p_GMM_y_prime += weights_prime[k] * multivariate_gaussian_pdf(alpha_estimate, means_prime[k],
                                                                              self.cluster_cov[y_prime][k])
            p_GMM_all.append(p_GMM_y_prime)

        # Compute numerator and denominator for each class
        numerator = p_GMM_all * prob_map  # Bayes' rule
        denominator = np.sum(numerator)

        should_keep_original = (
                np.any(np.array(p_GMM_all) == 0) or
                denominator < 1e-6
        )

        if should_keep_original:
            return prob_map, np.max(p_GMM_all)
        else:
            return (numerator / denominator).reshape(prob_map.shape), np.max(p_GMM_all)

    def get_cluster_parameters(self, class_idx):
        return self.cluster_centers.get(class_idx), self.mixture_weights.get(class_idx)


def multivariate_gaussian_pdf(x, mean, cov):
    K = len(x)
    x_m = x - mean
    try:
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(K) * 1e-6
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

    norm_factor = 1.0 / (np.power((2 * np.pi), K / 2) * np.power(det_cov, 0.5))
    exponent = -0.5 * np.dot(x_m.T, np.dot(inv_cov, x_m))
    return norm_factor * np.exp(exponent)


def load_models(data_name, subj, time_length, base_dir):
    """Load pretrained SincAdaptNet models for a given subject."""
    path = os.path.join(base_dir, f"sub{subj}", f"{time_length}s")
    ckpts = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.startswith('kf') and f.endswith('_model.pkl')
    )
    ensemble = []
    dp = get_dataset_params(data_name)
    for ckpt in ckpts:
        net = get_feature_extractor(data_name, dp['chn'], dp['sample_rate'])
        net.load_state_dict(torch.load(ckpt))
        net.eval().cuda()
        ensemble.append(net)
    return ensemble


def run_online_adaptation(ensemble, dataset_test, args, params):
    """Perform online BTTA-DG adaptation on test dataset."""
    loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    estimator = DirichletEstimator()
    clusterer = OnlineClustererGMM(
        num_components=params['num_components'],
        num_classes=args.class_num,
        conf_threshold=params['conf_threshold'],
        entropy_threshold=params['entropy_threshold']
    )
    acc_adapt = []
    for eeg_data, target in tqdm(loader, desc="Adapting..."):
        eeg_data, target = eeg_data.cuda(), target.cuda()
        # ensemble inference
        prob_maps, feats = [], []
        with torch.no_grad():
            for model in ensemble:
                emb, logits = EEGModel(model)(eeg_data)
                p_map = F.softmax(logits, dim=-1).cpu().numpy().squeeze()
                feats.append(F.softmax(emb, dim=1).cpu().numpy().squeeze())
                prob_maps.append(p_map)
        prob_map = np.mean(prob_maps, axis=0)
        feat = np.mean(feats, axis=0)
        pred = np.argmax(prob_map)
        # Dirichlet estimation
        alpha = estimator.estimate(feat)
        clusterer.update(alpha, pred, prob_map)
        calibrated, _ = clusterer.predict_class(alpha, prob_map)
        y_hat = np.argmax(calibrated)
        acc_adapt.append((y_hat == target.item()))
    return np.mean(acc_adapt)


def EA(x):
    from scipy.linalg import fractional_matrix_power
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def get_dataset_params(data_name):
    dataset_params = {
        'BNCI2014001': {'paradigm': 'MI', 'N': 9, 'chn': 22, 'class_num': 2, 'time_sample_num': 1001,
                        'sample_rate': 250, 'trial_num': 144, 'time_length': 4},
        'BNCI2014002': {'paradigm': 'MI', 'N': 14, 'chn': 15, 'class_num': 2, 'time_sample_num': 2561,
                        'sample_rate': 512, 'trial_num': 100, 'time_length': 5},
        'BNCI2015001': {'paradigm': 'MI', 'N': 12, 'chn': 13, 'class_num': 2, 'time_sample_num': 2561,
                        'sample_rate': 512, 'trial_num': 200, 'time_length': 5},
    }
    return dataset_params.get(data_name, None)


def get_feature_extractor(data_name, chn, sample_rate):
    fir_length = 51 if sample_rate == 250 else 51 * 2
    return SincAdaptNet(input_channel=chn, n_classes=2, n_spatial=16, n_fir=24, fir_length=fir_length,
                        batch_norm_alpha=0.1, fs=sample_rate)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    logging.basicConfig(level=logging.INFO)

    data_name = 'BNCI2014001'  # or BNCI2014002, BNCI2015001
    base_dir = f'./baseline_SincAdaptNet_{data_name}'
    params = {
        'BNCI2014001': {'num_components': 6, 'conf_threshold': 0.596, 'entropy_threshold': 0.673},
        'BNCI2014002': {'num_components': 4, 'conf_threshold': 0.515, 'entropy_threshold': 0.694},
        'BNCI2015001': {'num_components': 5, 'conf_threshold': 0.504, 'entropy_threshold': 0.689}
    }[data_name]

    dataset_params = get_dataset_params(data_name)
    results = []
    for subj in range(dataset_params['N']):
        # load data
        from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_cross_session
        import argparse
        args = argparse.Namespace(**dataset_params, idt=subj, data=data_name, method='')
        args.cross_session = False  # or "cross_session"
        if args.cross_session:
            train_x, train_y, test_x, test_y = read_mi_combine_tar_cross_session(args)
        else:
            train_x, train_y, test_x, test_y = read_mi_combine_tar(args)
        dataset_test = TensorDataset(torch.from_numpy(EA(test_x).astype(np.float32)).unsqueeze(1),
                                     torch.from_numpy(test_y).long())
        # load ensemble
        ensemble = load_models(data_name, subj, dataset_params['time_length'], base_dir)
        # adapt
        acc = run_online_adaptation(ensemble, dataset_test, args, params)
        results.append({'Subject': subj + 1, 'Accuracy': acc})
        logging.info(f"Subject {subj}: Adapted Acc = {acc:.4f}")

    df = pd.DataFrame(results)
    suffix = "_cross_session" if args.cross_session else ""
    df.to_excel(f'BTTA-DG_{data_name}{suffix}.xlsx', index=False)

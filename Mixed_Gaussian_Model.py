import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma



def get_multivariate_normal_probability(x:np.ndarray, mu:np.ndarray, sigma:np.ndarray|None=None):
    """
    x:     (n_samples,  n_features)
    mu:    (n_features)
    sigma: (n_features, n_features)
    """
    
    if x.ndim != 2 or mu.ndim != 1 or (sigma is not None and sigma.ndim != 2):
        raise ValueError("Shape of x or mu or sigma is incorrect")
    
    n_samples  = x.shape[0]
    n_features = x.shape[1]
    n_clusters = mu.shape[0]
    
    if len(mu) != n_features:
        raise ValueError("Shape of mu is incorrect")
    
    if sigma is not None and (sigma.shape[0] != n_features or sigma.shape[1] != n_features):
        raise ValueError("Shape of sigma is incorrect")
    
    if sigma is None:
        sigma = np.identity(n_features)
    
    prob = stats.multivariate_normal.pdf(x, mean=mu, cov=sigma)
    return prob


class My_K_means:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-7, random_state=None) -> None:
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        self.centers    = np.zeros((n_clusters, 1))
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
            
    def fit(self, data:np.ndarray) -> None:
        # エラー処理
        if type(data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        
        if data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        
        # 初期化
        num, dim     = data.shape
        data_labels  = self.random.integers(0, self.n_clusters, size=len(data))
        cluster_size = np.array([np.sum(data_labels == i)                                                 for i in range(0, self.n_clusters)])
        centers      = np.array([np.sum(data[np.where(data_labels == i)[0], :], axis=0) / cluster_size[i] for i in range(0, self.n_clusters)])
        
        # 最近傍法によるクラスタリング
        prev_error = 0
        for idx in range(0, self.max_iter):
            
            # クラスタリングの更新
            data_labels  = np.array([np.argmin(np.sum(np.square(data[idx2, :] - centers), axis=1)) for idx2 in range(0, len(data))])
            
            # クラスタの中心を更新                    
            cluster_size = np.array([np.sum(data_labels == i)                                                                for i in range(0, self.n_clusters)])
            centers      = np.array([np.sum(data[np.where(data_labels == i)[0], :], axis=0) / np.maximum(cluster_size[i], 1) for i in range(0, self.n_clusters)])
            
            # 誤差の計算
            error_amount = np.sum([np.square(data[i, :] - centers[data_labels[i], :]) for i in range(0, len(data))])
            
            # 収束条件の確認
            if np.abs(error_amount - prev_error) < self.tol:
                break
            else:
                prev_error = error_amount
        
        
        # クラスタリング結果の保存
        self.cluster_size     = cluster_size
        self.centers          = centers
        self.data_labels      = data_labels
        
        return None
    
    def predict(self, test_data:np.ndarray) -> np.ndarray:
        # エラー処理
        if type(test_data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(test_data) == 0:
            raise ValueError("Data must not be empty")
        
        if test_data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        
        # テストデータのクラスタリング
        test_labels = np.array([np.argmin(np.sum(np.square(test_data[idx, :] - self.centers), axis=1)) for idx in range(0, len(test_data))])
        
        return test_labels
        

class My_Mixed_Gaussian_Model_in_VB: # Variational Bayes
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-7, random_state=None) -> None:
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
        
    def fit(self, data:np.ndarray) -> None:
        # エラー処理
        if type(data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        
        if data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # パラメータの初期設定
        num, dim         = data.shape
        self.μ0          = 0 * np.ones(dim)
        self.σ0          = 1
        self.label_d_k   = self.random.random((num, self.n_clusters))
        self.cluster_μ_k = self.random.random((self.n_clusters, dim))
        self.cluster_σ_k = self.random.random((self.n_clusters, dim)) + 1e-5
        
        # 変分EMアルゴリズムの実行
        for idx in range(0, self.max_iter):
            # E-step: 負担率の計算
            z_dk = np.array([get_multivariate_normal_probability(data, self.cluster_μ_k[k, :]) for k in range(0, self.n_clusters)]).T
            z_dk = z_dk / np.sum(z_dk, axis=1, keepdims=True)
            
            
            # M-step: パラメータの更新(平均値μ, 分散値σ)
            # クラスターの平均値μの更新
            denominator = (np.sum(z_dk, axis=0, keepdims=True) + np.square(1/self.σ0)).T
            new_μ_k     = np.array([np.sum(z_dk[:, k].reshape((num, 1)) * data, axis=0) + np.square(1/self.σ0) * self.μ0 for k in range(0, self.n_clusters)])
            new_μ_k     = new_μ_k / denominator
            new_σ_k     = 1 / denominator
            
            error = np.sum(np.square(new_μ_k - self.cluster_μ_k)) + np.sum(np.square(new_σ_k - self.cluster_σ_k))
            if idx % 10 == 0:
                print(f"学習回数：{idx}")
                print(f"誤差：{error}")
                print()
            
            # 分布の更新
            self.label_d_k   = z_dk
            self.cluster_μ_k = new_μ_k
            self.cluster_σ_k = new_σ_k
            
            # 終了条件
            if error < self.tol:
                break
    
    def predict(self, test_data:np.ndarray) -> np.ndarray:
        # エラー処理
        if type(test_data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(test_data) == 0:
            raise ValueError("Data must not be empty")
        
        if test_data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # テストデータのクラスタリング
        test_labels = np.array([np.argmin(np.sum(np.square(test_data[idx, :] - self.cluster_μ_k), axis=1)) for idx in range(0, len(test_data))])
        
        return test_labels


class My_Mixed_Gaussian_Model_in_GS: # Gibbs Sampling
    def __init__(self, n_clusters=3, max_sampling=5000, burn_in=100, thinning=10, random_state=None) -> None:
        self.n_clusters   = n_clusters
        self.max_sampling = max_sampling
        self.burn_in      = burn_in
        self.thinning     = thinning
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
    def get_sampling_cluster_label(self, data:np.ndarray, cluster:np.ndarray) -> np.int64:
        # クラスタラベルのshapeを取得
        K, dim = cluster.shape
        
        # クラスタラベルの確率分布を取得
        sampling_prob = np.array([get_multivariate_normal_probability(data, self.cluster_μ_k[k, :]) for k in range(0, self.n_clusters)]).T
        sampling_prob = sampling_prob / np.sum(sampling_prob, axis=1, keepdims=True)
        
        # カテゴリカルサンプリング
        rnd_c = np.array([self.random.multinomial(n=1, pvals=sampling_prob[i, :], size=1)[0, :] for i in range(0, len(sampling_prob))])
        rnd_l = np.where(rnd_c == 1)
        return rnd_l[1]
    
    def fit(self, data:np.ndarray) -> None:
        # エラー処理
        if type(data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        
        if data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # パラメータの初期設定
        num, dim            = data.shape
        self.μ0             = 0 * np.ones(dim)
        self.σ0             = 1
        self.cluster_μ_k    = self.random.random((self.n_clusters, dim))
        self.sampling_label = np.array([0 for _ in range(0, num)]).reshape(num, 1)
        self.sampling_μ     = np.array([0 for _ in range(0, self.n_clusters) for _ in range(0, dim)]).reshape(self.n_clusters, dim, 1)
        
        # サンプル数の数だけループ
        for idx in range(0, self.max_sampling):
            # E-step: クラスタラベルのサンプリング
            z_d = self.get_sampling_cluster_label(data, self.cluster_μ_k)
            
            
            # M-step: パラメータの更新(平均値μ, 分散値σ)
            # クラスターの平均値μの更新
            denominator = np.array([(np.count_nonzero(z_d == k) + np.square(1/self.σ0)) for k in range(0, self.n_clusters)])
            new_μ_k     = np.array([np.sum(data[z_d == k, :], axis=0) + np.square(1/self.σ0) * self.μ0 for k in range(0, self.n_clusters)])
            new_μ_k     = new_μ_k / denominator.reshape((self.n_clusters, 1))
            
            # 進行状況の表示
            if idx % 100 == 0:
                print(f"サンプリング回数：{idx}")
                print()
            
            # 分布の更新
            self.cluster_μ_k = new_μ_k
            
            # サンプルデータの保存
            self.sampling_label = np.concatenate([self.sampling_label, z_d.reshape(num, 1)],                      axis=1)
            self.sampling_μ     = np.concatenate([self.sampling_μ,     new_μ_k.reshape(self.n_clusters, dim, 1)], axis=2)
        
        # 初期サンプルデータの削除
        self.sampling_label = self.sampling_label[:, 1:]
        self.sampling_μ     = self.sampling_μ[:, :, 1:]
        
        # 分布の更新
        self.cluster_μ_k = np.mean(self.sampling_μ[:, :, ::self.thinning], axis=2)
        
        
    
    def predict(self, test_data:np.ndarray) -> np.ndarray:
        # エラー処理
        if type(test_data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(test_data) == 0:
            raise ValueError("Data must not be empty")
        
        if test_data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # テストデータのクラスタリング
        test_labels = np.array([np.argmin(np.sum(np.square(test_data[idx, :] - self.cluster_μ_k), axis=1)) for idx in range(0, len(test_data))])
        
        return test_labels


class My_Mixed_Gaussian_Model_with_Variance_in_VB: # Variational Bayes
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-7, random_state=None) -> None:
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
        
    def fit(self, data:np.ndarray) -> None:
        # エラー処理
        if type(data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        
        if data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # パラメータの初期設定
        num, dim         = data.shape
        self.μ0          = 0 * np.ones(dim)
        self.σ0          = 1
        self.label_d_k   = self.random.random((num, self.n_clusters))
        self.cluster_μ_k = self.random.random((self.n_clusters, dim))
        self.cluster_σ_k = self.random.random((self.n_clusters, dim)) + 1e-5
        
        self.τa = self.random.random()
        self.τb = self.random.random()
        self.πk = self.random.random(self.n_clusters)
        
        
        # 変分EMアルゴリズムの実行
        for idx in range(0, self.max_iter):
            # 事後ラベル分布 パラメータの更新
            Σ    = (self.τb / self.τa) * np.identity(dim)
            z_dk = np.array([get_multivariate_normal_probability(data, self.cluster_μ_k[k, :], Σ) for k in range(0, self.n_clusters)]).T
            z_dk = z_dk * np.exp(digamma(self.πk))
            z_dk = z_dk / np.sum(z_dk, axis=1, keepdims=True)
            
            
            # 事後精度分布 パラメータの更新
            new_τ_a = num * dim / 2 + self.τa
            new_τ_b = (np.sum([np.dot(data[i,:], data[i,:]) for i in range(0, num)]) + self.n_clusters * self.σ0 * np.dot(self.μ0, self.μ0)) / 2 + self.τb
            
            
            # 事後クラスター中心分布 パラメータの更新(平均値μ, 分散値σ)
            # クラスターの平均値μの更新
            denominator = (np.sum(z_dk, axis=0, keepdims=True) + self.σ0).T
            new_μ_k     = np.array([np.sum(z_dk[:, k].reshape((num, 1)) * data, axis=0) + self.σ0 * self.μ0 for k in range(0, self.n_clusters)])
            new_μ_k     = new_μ_k / denominator
            new_σ_k     = 1 / ((new_τ_a / new_τ_b) * denominator)
            
            
            # 事後クラスタ分布 パラメータの更新
            new_πk = np.sum(z_dk, axis=0) + self.πk
            
            
            # 更新量の大きさを計算
            error_z_dk = np.sum(np.square(z_dk    - self.label_d_k))
            error_μ_k  = np.sum(np.square(new_μ_k - self.cluster_μ_k))
            error_σ_k  = np.sum(np.square(new_σ_k - self.cluster_σ_k))
            error_τa   = np.sum(np.square(new_τ_a - self.τa))
            error_τb   = np.sum(np.square(new_τ_b - self.τb))
            error_πk   = np.sum(np.square(new_πk  - self.πk))
            
            
            error = error_z_dk + error_μ_k + error_σ_k + error_τa + error_τb + error_πk
            if idx % 10 == 0:
                print(f"学習回数：{idx}")
                print(f"誤差：   {error}")
                print()
            
            # 分布の更新
            self.label_d_k   = z_dk
            self.cluster_μ_k = new_μ_k
            self.cluster_σ_k = new_σ_k
            self.τa          = new_τ_a
            self.τb          = new_τ_b
            self.πk          = new_πk
            
            # 終了条件
            if error < self.tol:
                break
    
    def predict(self, test_data:np.ndarray) -> np.ndarray:
        # エラー処理
        if type(test_data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(test_data) == 0:
            raise ValueError("Data must not be empty")
        
        if test_data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # テストデータのクラスタリング
        test_labels = np.array([np.argmin(np.sum(np.square(test_data[idx, :] - self.cluster_μ_k), axis=1)) for idx in range(0, len(test_data))])
        
        return test_labels


class My_Mixed_Gaussian_Model_with_Variance_in_GS: # Gibbs Sampling
    def __init__(self, n_clusters=3, max_sampling=5000, burn_in=100, thinning=10, random_state=None) -> None:
        self.n_clusters   = n_clusters
        self.max_sampling = max_sampling
        self.burn_in      = burn_in
        self.thinning     = thinning
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
    
    def get_sampling_label(self, data:np.ndarray, sample_μ:np.ndarray, sample_τ:np.ndarray, sample_π:np.ndarray) -> np.ndarray:
        # クラスタラベルのshapeを取得
        K, dim = sample_μ.shape
        
        # 分散共分散行列の設定
        sigma = np.identity(dim) / sample_τ
        
        # クラスタラベルの確率分布を取得
        sampling_prob = np.array([get_multivariate_normal_probability(data, sample_μ[k, :], sigma) * sample_π[k] for k in range(0, self.n_clusters)]).T
        sampling_prob = sampling_prob / np.sum(sampling_prob, axis=1, keepdims=True)
        
        # カテゴリカルサンプリング
        rnd_c = np.array([self.random.multinomial(n=1, pvals=sampling_prob[i, :], size=1)[0, :] for i in range(0, len(sampling_prob))])
        rnd_l = np.where(rnd_c == 1)
        return rnd_l[1]
    
    def get_sampling_cluster_centric(self, K:int, data:np.ndarray, sample_z:np.ndarray, sample_τ:np.ndarray, μ0:np.ndarray, ρ0:float):
        _, dim      = data.shape
        denominator = np.array([len(np.where(sample_z == k)[0]) + ρ0 for k in range(0, K)])
        
        mu  = np.array([(np.sum(data[np.where(sample_z == k)[0], :], axis=0) + ρ0 * μ0) / denominator[k] for k in range(0, K)])
        cov = np.array([np.identity(dim) / (sample_τ * denominator[k]) for k in range(0, K)])
        
        rnd_μ = np.array([self.random.multivariate_normal(mean=mu[k, :], cov=cov[k, :, :], check_valid='raise') for k in range(0, K)])
        return rnd_μ
    
    def get_sampling_cluster_variance(self, K:int, data:np.ndarray, sample_z:np.ndarray, μ0:np.ndarray, ρ0:float, α0:float, β0:float):
        num, dim = data.shape
        α        = num * dim / 2 + α0
        β        = np.sum(np.sum(np.square(data)) + K * ρ0 * np.dot(μ0, μ0))
        β        = β / 2 + β0
        
        rnd_τ = self.random.gamma(α, 1/β)
        return rnd_τ
    
    def get_sampling_cluster_label(self, K:int, sample_z:np.ndarray, a:float):
        hat_a = np.array([len(np.where(sample_z == k)[0]) + a for k in range(0, K)])
        
        rnd_π = self.random.dirichlet(alpha=hat_a)
        return rnd_π
    
    def fit(self, data:np.ndarray) -> None:
        # エラー処理
        if type(data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        
        if data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        
        # パラメータの初期設定
        num, dim        = data.shape
        self.μ0         = 0 * np.ones(dim)
        self.ρ0         = 1
        self.α0         = 1
        self.β0         = 1
        self.a          = 1
        self.sampling_z = np.array([0 for _ in range(0, num)]).reshape(num, 1)
        self.sampling_μ = np.array([0 for _ in range(0, self.n_clusters) for _ in range(0, dim)]).reshape(self.n_clusters, dim, 1)
        self.sampling_τ = np.array([0])
        self.sampling_π = np.array([0 for _ in range(0, self.n_clusters)]).reshape(self.n_clusters, 1)
        
        # 各サンプリング値の初期値
        new_z = np.array([0 for _ in range(0, num)])
        new_μ = self.random.random((self.n_clusters, dim))
        new_τ = self.random.random()
        new_π = self.random.random(self.n_clusters)
        
        
        # サンプル数の数だけループ
        for idx in range(0, self.max_sampling):
            
            # クラスタラベルのサンプリング
            new_z = self.get_sampling_label(data, new_μ, new_τ, new_π)
            
            # 事後クラスター中心分布のサンプリング
            new_μ = self.get_sampling_cluster_centric(self.n_clusters, data, new_z, new_τ, self.μ0, self.ρ0)
            
            # 事後クラスター分散分布のサンプリング
            new_τ = self.get_sampling_cluster_variance(self.n_clusters, data, new_z, self.μ0, self.ρ0, self.α0, self.β0)
            
            # 事後クラスターラベル分布のサンプリング
            new_π = self.get_sampling_cluster_label(self.n_clusters, new_z, self.a)
            
            
            # 進行状況の表示
            if idx % 100 == 0:
                print(f"サンプリング回数：{idx}")
                print()
            
            
            # サンプルデータの保存
            self.sampling_z = np.concatenate([self.sampling_z, new_z.reshape(num, 1)],                  axis=1)
            self.sampling_μ = np.concatenate([self.sampling_μ, new_μ.reshape(self.n_clusters, dim, 1)], axis=2)
            self.sampling_τ = np.concatenate([self.sampling_τ, [new_τ]],                                axis=0)
            self.sampling_π = np.concatenate([self.sampling_π, new_π.reshape(self.n_clusters, 1)],      axis=1)
        
        # 初期サンプルデータの削除
        self.sampling_z = self.sampling_z[:, 1:]
        self.sampling_μ = self.sampling_μ[:, :, 1:]
        self.sampling_τ = self.sampling_τ[1:]
        self.sampling_π = self.sampling_π[:, 1:]
        
        # 分布の更新
        self.cluster_μ_k = np.mean(self.sampling_μ[:, :, ::self.thinning], axis=2)
        
    
    def predict(self, test_data:np.ndarray) -> np.ndarray:
        # エラー処理
        if type(test_data) is not np.ndarray:
            raise TypeError("Data must be a numpy array")
        
        if len(test_data) == 0:
            raise ValueError("Data must not be empty")
        
        if test_data.ndim != 2:
            raise ValueError("The number of dimensions must be 2")
        
        # テストデータのクラスタリング
        test_labels = np.array([np.argmin(np.sum(np.square(test_data[idx, :] - self.cluster_μ_k), axis=1)) for idx in range(0, len(test_data))])
        
        return test_labels





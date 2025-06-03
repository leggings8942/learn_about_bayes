import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma

def modified_cholesky(x):
    if type(x) is list:
        x = np.array(x)
        
    if x.ndim != 2:
        print(f"x dims = {x.ndim}")
        raise ValueError("エラー：：次元数が一致しません。")
    
    if x.shape[0] != x.shape[1]:
        print(f"x shape = {x.shape}")
        raise ValueError("エラー：：正方行列ではありません。")
    
    n = x.shape[0]
    d = np.diag(x).copy()
    L = np.tril(x, k=-1).copy() + np.identity(n)
    
    for idx1 in range(1, n):
        prev = idx1 - 1
        tmp  = d[0:prev] if d[0:prev].size != 0 else 0
        tmp  = np.dot(L[idx1:, 0:prev], (L[prev, 0:prev] * tmp).T)
        
        DIV  = d[prev] if d[prev] != 0 else 1e-16
        L[idx1:, prev] = (L[idx1:, prev] - tmp) / DIV
        d[idx1]       -= np.sum((L[idx1, 0:idx1] ** 2) * d[0:idx1])
    
    d = np.diag(d)
    return L, d

def log_likelihood_of_normal_distrubution(x, mean, cov):
    assert x.shape == mean.shape,        f"argument sizes do not match:: x.shape = {x.shape}, mean.shape = {mean.shape}"
    assert cov.shape[0] == cov.shape[1], f"covariance matrix must be square:: cov.shape[0] = {cov.shape[0]}, cov.shape[1] = {cov.shape[1]}"
    assert x.shape[0] == cov.shape[0],   f"number of dimensions of convariance matrix and input vector do not match:: x.shape[0] = {x.shape[0]}, cov.shape[0] = {cov.shape[0]}"
    
    try:
        diff  = x - mean
        sigma = np.dot(np.linalg.pinv(cov), diff)
    except Exception as e:
        try:
            sigma = np.linalg.solve(cov, diff)
        except Exception as e:
            sigma = np.zeros_like(diff)
    finally:
        mult  = np.dot(diff.T, sigma)
        mult  = np.abs(mult)
        mult  = np.diag(mult)
    
    d              = x.shape[0]
    log_likelihood = d * np.log(2 * np.pi) + np.log(np.abs(np.linalg.det(cov)) + 1e-256) + mult
    return -log_likelihood / 2

# 軟判別閾値関数
def soft_threshold(x, α):
    return np.sign(x) * np.maximum(np.abs(x) - α, 0)



class Update_Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999):
        self.alpha   = alpha
        self.beta1   = beta1
        self.beta2   = beta2
        self.time    = 0
        self.beta1_t = 1
        self.beta2_t = 1
        self.m = np.array([])
        self.v = np.array([])

    def update(self, grads):
        if self.time == 0:
            self.m = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)
        
        ε = 1e-32
        self.time   += 1
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1_t)
        v_hat = self.v / (1 - self.beta2_t)
        
        output = self.alpha * m_hat / np.sqrt(v_hat + ε)
        return output


def soft_maximum(x, α):
    sign = np.empty_like(x)
    sign[x >= 0] =  1
    sign[x <  0] = -1
    return sign * (np.abs(x) + α)

class Update_Rafael:
    def __init__(self, alpha=0.01, beta=0.99, isSHC=False):
        self.alpha  = alpha
        self.beta   = beta
        self.time   = 0
        self.beta_t = 1
        self.m = np.array([])
        self.v = np.array([])
        self.w = np.array([])
        self.σ_coef = 0
        self.isSHC = isSHC

    def update(self, grads):
        if self.time == 0:
            self.m = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)
            self.w = np.zeros(grads.shape)
            self.σ_coef = (1 + self.beta) / 2
        
        ε = 1e-32
        self.time   += 1
        self.beta_t *= self.beta

        self.m = self.beta * self.m + (1 - self.beta) * grads
        m_hat = self.m / (1 - self.beta_t)

        self.v = self.beta * self.v + (1 - self.beta) * (grads ** 2)
        self.w = self.beta * self.w + (1 - self.beta) * ((grads / soft_maximum(m_hat, ε) - 1) ** 2)
        
        if self.beta - self.beta_t > 0.1:
            v_hat  = self.v * self.σ_coef / (self.beta - self.beta_t)
            w_hat  = self.w * self.σ_coef / (self.beta - self.beta_t)
            σ_com  = np.sqrt((v_hat + w_hat + ε) / 2)
            # σ_hes  = np.sqrt(w_hat + ε)
            
            # self-healing canonicalization
            R = 0
            if self.isSHC:
                def chebyshev(r):
                    tmp1 = σ_com + r
                    tmp2 = np.square(m_hat / tmp1)
                    f    =     np.sum(tmp2,                   axis=0) - r
                    df   = 2 * np.sum(tmp2 / tmp1,            axis=0) + 1
                    ddf  = 6 * np.sum(tmp2 / np.square(tmp1), axis=0)
                    newt = f / df
                    return r + newt + ddf / (2 * df) * np.square(newt)
                
                r_min = np.sum(np.square(m_hat / σ_com), axis=0)
                r_max = np.cbrt(np.sum(np.square(m_hat), axis=0))
                R = np.maximum(np.minimum(r_max, r_min), 1)
                R = chebyshev(R)
                # R = chebyshev(R)     # option: 精度を求めるならチェビシェフ法を2回適用する
                R = np.maximum(R, 1) # option: 収束速度は遅くなるが、安定性が向上する
                
            output = self.alpha * m_hat / (σ_com + R)
        else:
            output = self.alpha * np.sign(grads)
        
        return output


        
class Bayesian_Auto_Regressive:
    def __init__(self, train_data, tol=1e-7, max_iterate=100000, random_state=None) -> None:
        if type(train_data) is pd.core.frame.DataFrame:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if type(train_data) is not np.ndarray:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if train_data.ndim != 1:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        self.train_data  = train_data
        self.lags        = 0
        self.tol         = tol
        self.max_iterate = max_iterate
        self.learn_flg   = False

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random
        
        return None

    def fit(self, lags:int=1, visible_flg:bool=False) -> bool:
        # caution!!!
        # OLS(Ordinary Learst Squares)推定量を計算する際に
        # 擬似逆行列(pinv関数)を使用している箇所が存在する
        # この処理は逆行列が存在しない場合(行列式が0の場合)に発火する
        # しかし理論的には逆行列が存在しない時系列データの組み合わせは極めて稀である(無視できる)
        # 入力された時系列データ自体にミスが存在する(0の定数列になっている等)可能性が高い
        # statsmodels.tsa.vector_ar.var_model.VARではこのような時系列データを入力として与えた場合には
        # エラーを出力するようになっている
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである

        nobs = len(self.train_data)
        if not nobs - lags - 1 > 0:
            # データ数に対して、最尤推定対象が多すぎる
            self.learn_flg = False
            return self.learn_flg
        
        # ===============================================================================
        # 学習変数の初期化
        # ===============================================================================
        x_data = np.array([self.train_data[t-lags : t][::-1] for t in range(lags, nobs)])
        y_data = self.train_data[lags:]
        
        self.A = self.random.random(size=(lags, 2))
        self.B = self.random.random(size=2)
        self.Σ = self.random.random(size=2)
        
        self.doc_num = len(y_data)
        self.lags    = lags
        self.k       = 1
        self.θ       = 1
        self.α       = 1
        self.β       = 1
        
        # ===============================================================================
        # 変分推定部
        # ===============================================================================
        num, s   = x_data.shape
        prev_Err = 0
        for idx in range(0, self.max_iterate):
            # 初期化
            Σ_new = np.zeros_like(self.Σ)
            
            Σ_new[0] = self.doc_num / 2 + self.α
            Σ_new[1] = self.β
            for d in range(0, self.doc_num):
                Σ_new[1] += (np.log(y_data[d])**2) / 2
                Σ_new[1] -= np.sum(self.A[:, 0] * self.A[:, 1] * x_data[d, :]) * np.log(y_data[d])
                Σ_new[1] -= (self.B[0] * self.B[1]) * np.log(y_data[d])
                Σ_new[1] += np.sum((self.A[:, 0]**2) * (self.A[:, 1] + 1) * self.A[:, 1] * (x_data[d, :]**2)) / 2
                Σ_new[1] += np.sum(x_data[d, l] * x_data[d, ld] * self.A[l, 0] * self.A[l, 1] * self.A[ld, 0] * self.A[ld, 1] for l in range(0, self.lags) for ld in range(l+1, self.lags))
                Σ_new[1] += (self.B[0] * self.B[1]) * np.sum(self.A[:, 0] * self.A[:, 1] * x_data[d, :])
                Σ_new[1] += (self.B[0]**2) * (self.B[1] + 1) * self.B[1] / 2
            
            
            # ===========================================================================
            # ブラックボックス変分推定部
            # ===========================================================================
            
            self.A = np.sqrt(2 * self.A)
            self.B = np.sqrt(2 * self.B)
            optimizer_A = Update_Rafael(0.001, isSHC=True)
            optimizer_B = Update_Rafael(0.001, isSHC=True)
            for idx2 in range(0, 2000):
                q_A = np.square(self.A) / 2
                q_B = np.square(self.B) / 2
                
                A_new = np.zeros_like(q_A)
                B_new = np.zeros_like(q_B)
                
                for d in range(0, self.max_iterate):
                    avg_Σ        = Σ_new[0] / Σ_new[1]
                    tmp1         = avg_Σ * x_data[d, :] * np.log(y_data[d])
                    tmp1        -= avg_Σ * x_data[d, :] * (q_B[0] * q_B[1])
                    tmp1        -= 1 / self.θ
                    A_new[:, 0] += tmp1 * q_A[:, 1]
                    A_new[:, 1] += tmp1 * q_A[:, 0]
                    
                    A_new[:, 0] -= avg_Σ * (x_data[d, :]**2) *  q_A[:, 0]     * (  q_A[:, 1]**2 + q_A[:, 1])
                    A_new[:, 1] -= avg_Σ * (x_data[d, :]**2) * (q_A[:, 0]**2) * (2*q_A[:, 1]    + 1)         / 2
                    
                    x_l          = x_data[d, :].reshape((self.lags, 1))
                    x_dl         = x_data[d, :].reshape((1, self.lags))
                    avg_A        = (q_A[:, 0] * q_A[:, 1]).reshape((1, self.lags))
                    A_lθ         = avg_Σ * (x_l * q_A[:, 1].reshape((self.lags, 1))) * (x_dl * avg_A)
                    A_lk         = avg_Σ * (x_l * q_A[:, 0].reshape((self.lags, 1))) * (x_dl * avg_A)
                    A_new[:, 0] -= np.sum(A_lθ, axis=1) - A_lθ
                    A_new[:, 1] -= np.sum(A_lk, axis=1) - A_lk
                    
                    A_new[:, 0] +=  self.k / q_A[:, 0]
                    A_new[:, 1] += (self.k - q_A[:, 1]) * polygamma(1, q_A[:, 1]) + 1
                    
                    
                    tmp1      = avg_Σ * np.log(y_data[d])
                    tmp1     -= avg_Σ * np.sum(q_A[:, 0] * q_A[:, 1] * x_data[d, :])
                    tmp1     -= 1 / self.θ
                    B_new[0] += tmp1 * q_B[:, 1]
                    B_new[1] += tmp1 * q_B[:, 0]
                    
                    B_new[0] -= avg_Σ *  q_B[:, 0]     * (q_B[:, 1]**2  + q_B[:, 1])
                    B_new[1] -= avg_Σ * (q_B[:, 0]**2) * (2 * q_B[:, 1] + 1)         / 2
                    
                    B_new[0] +=  self.k / q_B[:, 0]
                    B_new[1] += (self.k - q_B[:, 1]) * polygamma(1, q_B[:, 1]) + 1
                
                # 勾配法適用
                A_diff = A_new * self.A
                Δdiff  = optimizer_A.update(A_diff)
                self.A = self.A + Δdiff
                
                B_diff = B_new * self.B
                Δdiff  = optimizer_B.update(B_diff)
                self.B = self.B + Δdiff
                
                A_sum_diff = np.sum(np.abs(A_diff))
                A_per_diff = np.sum(np.abs(A_diff)) / (A_diff.size)
                B_sum_diff = np.sum(np.abs(B_diff))
                B_per_diff = np.sum(np.abs(B_diff)) / (B_diff.size)
                sum_diff = A_sum_diff + B_sum_diff
                per_diff = A_per_diff + B_per_diff
                if visible_flg and (idx2 % 100 == 0):
                    line = f'idx:{idx} BB変分部: idx2:{idx2} '
                    print(line          + f' 総微分量：{sum_diff}')
                    print(' '*len(line) + f' 要素あたりの微分量：{per_diff}')
                    print(' '*len(line) + f' q(A): 総微分量:{A_sum_diff}')
                    print(' '*len(line) + f' q(A): 要素あたりの微分量:{A_per_diff}')
                    print(' '*len(line) + f' q(B): 装備分量:{B_sum_diff}')
                    print(' '*len(line) + f' q(B): 要素あたりの微分量:{B_per_diff}')
                
                if sum_diff < 0.1:
                    break
            
            # 変数変換
            self.A = np.square(self.A) / 2
            self.B = np.square(self.B) / 2
            
            # デバッグ出力
            error_Σ  = np.sum(np.abs(self.Σ - Σ_new))
            diff_err = np.abs(error_Σ - prev_Err)
            if visible_flg and (idx % 1 == 0):
                print(f'学習回数:{idx}')
                print(f'q(Σ): 総誤差量:{error_Σ}')
                print(f'q(Σ): 前回からの誤差の変化量:{diff_err}')
            
            # 分散分布の更新
            self.Σ = Σ_new
            
            # 終了条件
            if diff_err <= self.tol:
                break
            else:
                prev_Err = error_Σ
        
        ValueError()
        
        self.learn_flg = True
        y_pred         = self.predict(x_data)
        self.sigma     = np.var(y_pred - y_data)
        self.data_num  = x_data.shape[0]

        return self.learn_flg

    def predict(self, test_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if type(test_data) is not np.ndarray:
            print(f"type(test_data) = {type(test_data)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        y_pred = np.sum(self.alpha * test_data, axis=1) + self.alpha0
        
        return y_pred
    
    def log_likelihood(self, test_data) -> np.float64:
        if type(test_data) is pd.core.series.Series:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if type(test_data) is not np.ndarray:
            print(f"type(test_data) = {type(test_data)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if test_data.ndim != 1:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise

        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise

        nobs   = len(test_data)
        x_data = np.array([test_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
        y_data = test_data[self.lags:]

        num, _ = x_data.shape
        y_pred = self.predict(x_data)

        prob           = np.frompyfunc(normal_distribution, 3, 1)(y_data, y_pred, np.sqrt(self.sigma))
        prob           = prob.astype(float).reshape([num, 1])
        log_likelihood = np.sum(np.log(prob + 1e-32))

        return log_likelihood
    
    def model_reliability(self, test_data, ic="aic") -> np.float64:
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        num, s = self.data_num, self.alpha.shape[0]
        log_likelihood = self.log_likelihood(test_data)

        inf = 0
        if ic == "aic":
            inf = -2 * log_likelihood + 2 * (s + 2)
        elif ic == "bic":
            inf = -2 * log_likelihood + (s + 2) * np.log(num)
        elif ic == "hqic":
            inf = -2 * log_likelihood + 2 * (s + 2) * np.log(np.log(num))
        else:
            raise

        return inf

    def select_order(self, train_data, maxlag=15, ic="aic", solver="normal equations", isVisible=False) -> int:
        if isVisible == True:
            print(f"AR model | {ic}", flush=True)
        
        nobs = len(train_data)
        if nobs <= maxlag:
            maxlag = nobs - 1

        model_param = []
        for lag in range(1, maxlag + 1):
            flg = self.fit(train_data, lags=lag, solver=solver)
            
            if flg:
                rel = self.model_reliability(train_data, ic=ic)
                model_param.append([rel, lag])
            else:
                rel = np.finfo(np.float64).max
                model_param.append([rel, lag])

            if isVisible == True:
                print(f"AR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag = np.finfo(np.float64).max, 0
        for elem in model_param:
            tmp_rel, tmp_lag = elem
            if res_rel > tmp_rel:
                res_rel    = tmp_rel
                res_lag    = tmp_lag
        
        res_lag = res_lag if res_lag != 0 else 1
        self.fit(train_data, lags=res_lag, solver=solver)
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了しませんでした。")
            raise
        
        if isVisible == True:
            print(f"selected orders | {res_lag}", flush=True)

        return self.lags

    def stat_inf(self):
        info = {}
        info["mean"]               = self.alpha0 / (1 - np.sum(self.alpha))
        info["variance"]           = self.sigma / (1 - np.sum(np.square(self.alpha)))
        info["standard deviation"] = np.sqrt(info["variance"])

        return info

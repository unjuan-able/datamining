import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix,
    roc_auc_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ========================= 配置区域 =========================
CONFIG = {
    # 数据路径
    'train_data_path': "your_data.csv",           # 训练集CSV
    'test_data_path': "其他目录/your_real_test_data.csv",  # 独立测试集CSV
    
    # 输出路径
    'scaler_path': "scaler.pkl",                  # 标准化器保存路径
    'model_path': "xgboost_model.json",           # 模型保存路径
    'augmented_data_path': "augmented_train.csv", # 增强后数据保存路径
    
    # 训练参数
    'test_size': 0.2,
    'random_state': 42,
    
    # XGBoost参数
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'early_stopping_rounds': 20,
    
    # 定向扩充参数
    'enable_augmentation': True,      # 是否启用定向扩充
    'badcase_analysis': True,         # 是否先做badcase分析
    'n_clusters': 3,                  # badcase聚类数
    'k_neighbors': 20,                # 检索相似样本数
    'augment_ratio': 5,               # 每个种子样本生成几个增强样本
    'noise_scale': 0.03,              # 特征噪声强度（相对值）
    'hard_sample_weight': 3.0,        # 困难样本过采样权重
}

# ========================= 1. 数据加载与预处理 =========================

def load_and_preprocess(data_path, scaler=None, fit_scaler=False):
    """
    加载CSV数据，第一列为标签，其余为特征
    返回: X, y, scaler
    """
    data = pd.read_csv(data_path)
    y = data.iloc[:, 0].values.astype(int)
    X = data.iloc[:, 1:].values.astype(np.float32)
    
    print(f"[数据] 加载: {data_path}, 样本数: {len(y)}, 特征数: {X.shape[1]}, 类别分布: {dict(Counter(y))}")
    
    if np.isnan(X).any():
        raise ValueError("数据包含 NaN，请先处理缺失值")
    
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, CONFIG['scaler_path'])
        print(f"[预处理] 标准化器已保存至: {CONFIG['scaler_path']}")
    elif scaler is not None:
        X = scaler.transform(X)
        print("[预处理] 使用已有标准化器转换数据")
    
    return X, y, scaler


# ========================= 2. Badcase分析与归因 =========================

class BadcaseAnalyzer:
    """
    分析模型预测错误，对badcase聚类，定位错误模式
    """
    def __init__(self, X_train, y_train, X_val, y_val, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        
        # 预测验证集
        self.y_pred = model.predict(X_val)
        self.y_prob = model.predict_proba(X_val)[:, 1]
        
        # 识别badcase索引
        self.fn_mask = (y_val == 1) & (self.y_pred == 0)  # 漏检：实际是欺诈，预测正常
        self.fp_mask = (y_val == 0) & (self.y_pred == 1)  # 误杀：实际正常，预测欺诈
        
        print(f"\n[Badcase分析] FN(漏检欺诈): {self.fn_mask.sum()}, FP(误杀正常): {self.fp_mask.sum()}")
    
    def cluster_badcases(self, n_clusters=3):
        """
        对badcase的embedding（特征+预测概率）做聚类，找出错误模式
        """
        badcase_mask = self.fn_mask | self.fp_mask
        if badcase_mask.sum() < n_clusters:
            print(f"[警告] Badcase数量({badcase_mask.sum()})少于聚类数({n_clusters})，跳过聚类")
            return None
        
        # 构造badcase特征：原始特征 + 预测概率（作为embedding）
        badcase_X = self.X_val[badcase_mask]
        badcase_prob = self.y_prob[badcase_mask].reshape(-1, 1)
        badcase_features = np.hstack([badcase_X, badcase_prob])
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(badcase_features)
        
        print(f"\n[聚类结果] Badcase分布:")
        for i in range(n_clusters):
            cluster_mask = labels == i
            fn_in_cluster = self.fn_mask[badcase_mask][cluster_mask].sum()
            fp_in_cluster = self.fp_mask[badcase_mask][cluster_mask].sum()
            print(f"  类别 {i}: 共{cluster_mask.sum()}个, FN={fn_in_cluster}, FP={fp_in_cluster}")
        
        # 返回每个聚类的中心特征（不含概率列）
        centroids = kmeans.cluster_centers_[:, :-1]
        return centroids, labels
    
    def get_error_samples(self, error_type='fn'):
        """
        获取指定错误类型的样本
        error_type: 'fn' (漏检), 'fp' (误杀), 'all' (全部)
        """
        if error_type == 'fn':
            return self.X_val[self.fn_mask], self.y_val[self.fn_mask]
        elif error_type == 'fp':
            return self.X_val[self.fp_mask], self.y_val[self.fp_mask]
        else:
            mask = self.fn_mask | self.fp_mask
            return self.X_val[mask], self.y_val[mask]


# ========================= 3. 训练集相似样本检索 =========================

class SimilarityRetriever:
    """
    在训练集中检索与目标样本（badcase）最相似的样本
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # 使用KNN构建索引（也可用FAISS加速大规模数据）
        self.knn = NearestNeighbors(n_neighbors=50, metric='euclidean', algorithm='auto')
        self.knn.fit(X_train)
    
    def retrieve(self, query_X, k=20, label_filter=None):
        """
        检索与query_X最相似的训练样本
        label_filter: 只返回指定标签的样本（如只找欺诈样本）
        """
        distances, indices = self.knn.kneighbors(query_X, n_neighbors=k)
        
        retrieved_samples = []
        retrieved_labels = []
        retrieved_indices = []
        
        for i, idx_list in enumerate(indices):
            for j, idx in enumerate(idx_list):
                if label_filter is not None and self.y_train[idx] != label_filter:
                    continue
                retrieved_samples.append(self.X_train[idx])
                retrieved_labels.append(self.y_train[idx])
                retrieved_indices.append(idx)
        
        if len(retrieved_samples) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(retrieved_samples), np.array(retrieved_labels), np.array(retrieved_indices)
    
    def retrieve_by_centroids(self, centroids, k=20, label_filter=None):
        """
        基于聚类中心检索相似样本
        """
        return self.retrieve(centroids, k=k, label_filter=label_filter)


# ========================= 4. 特征空间定向扩充 =========================

class FeatureSpaceAugmentor:
    """
    在特征空间对样本做定向增强
    """
    def __init__(self, noise_scale=0.03, augment_ratio=5):
        self.noise_scale = noise_scale
        self.augment_ratio = augment_ratio
    
    def gaussian_noise(self, X):
        """
        加高斯噪声：模拟特征测量误差
        noise_scale是相对值，基于特征标准差
        """
        X_aug = X.copy()
        feat_std = X.std(axis=0) + 1e-8
        noise = np.random.normal(0, self.noise_scale * feat_std, X.shape)
        return X_aug + noise
    
    def feature_interpolation(self, X, X_target, alpha_range=(0.3, 0.7)):
        """
        向目标样本方向插值：让样本更靠近badcase的决策区域
        """
        alphas = np.random.uniform(*alpha_range, size=(X.shape[0], 1))
        return alphas * X_target + (1 - alphas) * X
    
    def feature_perturbation(self, X, important_indices=None, perturb_ratio=0.1):
        """
        对重要特征做扰动：如果知道哪些特征重要，可以针对性扰动
        """
        X_aug = X.copy()
        if important_indices is None:
            # 随机选择部分特征做扰动
            n_features = X.shape[1]
            n_perturb = max(1, int(n_features * perturb_ratio))
            important_indices = np.random.choice(n_features, n_perturb, replace=False)
        
        feat_std = X[:, important_indices].std(axis=0) + 1e-8
        noise = np.random.normal(0, self.noise_scale * 2 * feat_std, (X.shape[0], len(important_indices)))
        X_aug[:, important_indices] += noise
        return X_aug
    
    def augment(self, X, y, X_target=None, mode='noise'):
        """
        对样本做增强
        mode: 'noise'(加噪), 'interpolate'(向目标插值), 'mixed'(混合)
        """
        X_aug_list = [X]  # 保留原始样本
        y_aug_list = [y]
        
        for _ in range(self.augment_ratio):
            if mode == 'noise':
                X_new = self.gaussian_noise(X)
            elif mode == 'interpolate' and X_target is not None:
                X_new = self.feature_interpolation(X, X_target)
            elif mode == 'mixed':
                X_new = self.gaussian_noise(X)
                if X_target is not None and np.random.rand() > 0.5:
                    X_new = self.feature_interpolation(X_new, X_target)
            else:
                X_new = self.gaussian_noise(X)
            
            X_aug_list.append(X_new)
            y_aug_list.append(y.copy())
        
        return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ========================= 5. 定向扩充主流程 =========================

def directed_augmentation_pipeline(X_train, y_train, X_val, y_val, model, config):
    """
    完整的定向扩充流程
    """
    print("\n" + "="*60)
    print("开始定向扩充流程")
    print("="*60)
    
    # Step 1: Badcase分析
    analyzer = BadcaseAnalyzer(X_train, y_train, X_val, y_val, model)
    
    # Step 2: Badcase聚类
    cluster_result = analyzer.cluster_badcases(n_clusters=config['n_clusters'])
    if cluster_result is None:
        print("[跳过] Badcase数量不足，无法进行定向扩充")
        return X_train, y_train, None
    
    centroids, badcase_labels = cluster_result
    
    # Step 3: 在训练集中检索相似样本
    retriever = SimilarityRetriever(X_train, y_train)
    
    # 分别处理FN和FP
    # FN（漏检欺诈）：需要更多"欺诈类"样本，在训练集中找与FN相似的欺诈样本
    fn_X, fn_y = analyzer.get_error_samples('fn')
    fp_X, fp_y = analyzer.get_error_samples('fp')
    
    seeds_fn = []
    seeds_fp = []
    
    # 对FN：在训练集中找标签=1（欺诈）且与FN相似的样本
    if len(fn_X) > 0:
        print(f"\n[检索] 针对FN({len(fn_X)}个)，在训练集中检索相似欺诈样本...")
        fn_centroids = KMeans(n_clusters=min(3, len(fn_X)), random_state=42).fit(fn_X).cluster_centers_
        sim_X, sim_y, sim_idx = retriever.retrieve_by_centroids(
            fn_centroids, k=config['k_neighbors'], label_filter=1
        )
        if len(sim_X) > 0:
            seeds_fn = sim_idx
            print(f"  -> 检索到 {len(np.unique(sim_idx))} 个欺诈类种子样本")
    
    # 对FP：在训练集中找标签=0（正常）且与FP相似的样本
    if len(fp_X) > 0:
        print(f"\n[检索] 针对FP({len(fp_X)}个)，在训练集中检索相似正常样本...")
        fp_centroids = KMeans(n_clusters=min(3, len(fp_X)), random_state=42).fit(fp_X).cluster_centers_
        sim_X, sim_y, sim_idx = retriever.retrieve_by_centroids(
            fp_centroids, k=config['k_neighbors'], label_filter=0
        )
        if len(sim_X) > 0:
            seeds_fp = sim_idx
            print(f"  -> 检索到 {len(np.unique(sim_idx))} 个正常类种子样本")
    
    # 合并种子索引（去重）
    all_seed_indices = np.unique(np.concatenate([
        seeds_fn if len(seeds_fn) > 0 else [],
        seeds_fp if len(seeds_fp) > 0 else []
    ])).astype(int)
    
    if len(all_seed_indices) == 0:
        print("[警告] 未检索到有效种子样本，跳过扩充")
        return X_train, y_train, None
    
    print(f"\n[种子] 共 {len(all_seed_indices)} 个种子样本用于定向扩充")
    
    # Step 4: 特征空间增强
    augmentor = FeatureSpaceAugmentor(
        noise_scale=config['noise_scale'],
        augment_ratio=config['augment_ratio']
    )
    
    X_augmented = [X_train]
    y_augmented = [y_train]
    
    # 对FN种子（欺诈类）做增强：向FN中心方向插值 + 噪声
    if len(seeds_fn) > 0:
        fn_seed_X = X_train[seeds_fn]
        fn_seed_y = y_train[seeds_fn]
        # 目标方向：FN样本中心（让欺诈样本更"像"漏检的欺诈）
        fn_target = fn_X.mean(axis=0) if len(fn_X) > 0 else fn_seed_X.mean(axis=0)
        
        X_fn_aug, y_fn_aug = augmentor.augment(
            fn_seed_X, fn_seed_y, X_target=fn_target.reshape(1, -1), mode='mixed'
        )
        X_augmented.append(X_fn_aug[1:])  # 去掉原始样本（已包含）
        y_augmented.append(y_fn_aug[1:])
        print(f"[增强] 欺诈类(FN)种子生成 {len(X_fn_aug)-1} 个新样本")
    
    # 对FP种子（正常类）做增强：向FP中心方向插值 + 噪声
    if len(seeds_fp) > 0:
        fp_seed_X = X_train[seeds_fp]
        fp_seed_y = y_train[seeds_fp]
        fp_target = fp_X.mean(axis=0) if len(fp_X) > 0 else fp_seed_X.mean(axis=0)
        
        X_fp_aug, y_fp_aug = augmentor.augment(
            fp_seed_X, fp_seed_y, X_target=fp_target.reshape(1, -1), mode='mixed'
        )
        X_augmented.append(X_fp_aug[1:])
        y_augmented.append(y_fp_aug[1:])
        print(f"[增强] 正常类(FP)种子生成 {len(X_fp_aug)-1} 个新样本")
    
    # 合并
    X_train_new = np.vstack(X_augmented)
    y_train_new = np.hstack(y_augmented)
    
    print(f"\n[扩充完成] 训练集: {X_train.shape[0]} -> {X_train_new.shape[0]} 样本")
    print(f"  类别分布: {dict(Counter(y_train_new))}")
    
    # 保存增强后的数据（可选）
    aug_df = pd.DataFrame(np.hstack([y_train_new.reshape(-1, 1), X_train_new]))
    aug_df.to_csv(config['augmented_data_path'], index=False)
    print(f"[保存] 增强数据已保存至: {config['augmented_data_path']}")
    
    # Step 5: 构建困难样本权重（用于WeightedRandomSampler或sample_weight）
    # 种子样本及其增强样本给予更高权重
    sample_weights = np.ones(len(y_train_new))
    seed_mask = np.zeros(len(y_train_new), dtype=bool)
    seed_mask[:len(y_train)] = True
    # 标记原始种子样本位置（近似）
    # 更精确的做法：记录增强样本的索引范围
    # 这里简化：对所有新生成样本给更高权重
    sample_weights[len(y_train):] = config['hard_sample_weight']
    
    return X_train_new, y_train_new, sample_weights


# ========================= 6. 模型训练与评估 =========================

def train_model(X_train, y_train, X_val, y_val, config, sample_weights=None):
    """
    训练XGBoost模型
    """
    # 计算类别权重
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if len(np.unique(y_train)) == 2 else 1
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',  # 用AUC做早停更稳定
        random_state=config['random_state'],
        scale_pos_weight=scale_pos_weight,
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        learning_rate=config['learning_rate'],
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1
    )
    
    # 如果提供了sample_weights，使用它
    fit_params = {
        'eval_set': [(X_val, y_val)],
        'verbose': False
    }
    if sample_weights is not None:
        fit_params['sample_weight'] = sample_weights[:len(X_train)]
    
    model.fit(X_train, y_train, **fit_params)
    
    # 如果早停触发了，输出最佳轮数
    if hasattr(model, 'best_iteration'):
        print(f"[训练] 最佳迭代轮数: {model.best_iteration}")
    
    return model


def evaluate_model(model, X_test, y_test, dataset_name="测试集"):
    """
    全面评估模型性能
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else None
    
    print(f"\n[{dataset_name} 评估结果]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    if auc is not None:
        print(f"  AUC      : {auc:.4f}")
    
    print(f"\n  混淆矩阵:")
    print(f"  {confusion_matrix(y_test, y_pred)}")
    
    return {
        'accuracy': acc, 'precision': prec, 
        'recall': rec, 'f1': f1, 'auc': auc
    }


# ========================= 7. 主流程 =========================

def main():
    config = CONFIG
    
    # ---- 7.1 加载训练数据 ----
    X, y, scaler = load_and_preprocess(config['train_data_path'], fit_scaler=True)
    
    # ---- 7.2 划分训练集和验证集（验证集用于badcase分析和早停）----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=y
    )
    print(f"\n[划分] 训练集: {len(y_train)}, 验证集: {len(y_val)}")
    
    # ---- 7.3 基线训练（第一次训练，用于badcase分析）----
    print("\n" + "="*60)
    print("Phase 1: 基线模型训练")
    print("="*60)
    
    baseline_model = train_model(X_train, y_train, X_val, y_val, config)
    baseline_metrics = evaluate_model(baseline_model, X_val, y_val, "验证集(基线)")
    
    # 保存基线模型
    baseline_model.save_model(config['model_path'].replace('.json', '_baseline.json'))
    
    # ---- 7.4 定向扩充（可选）----
    if config['enable_augmentation'] and config['badcase_analysis']:
        X_train_aug, y_train_aug, sample_weights = directed_augmentation_pipeline(
            X_train, y_train, X_val, y_val, baseline_model, config
        )
        
        # ---- 7.5 使用扩充数据重训练 ----
        print("\n" + "="*60)
        print("Phase 2: 定向扩充后重训练")
        print("="*60)
        
        final_model = train_model(X_train_aug, y_train_aug, X_val, y_val, config, sample_weights)
        
        # 验证集评估
        aug_metrics = evaluate_model(final_model, X_val, y_val, "验证集(扩充后)")
        
        # 对比
        print("\n[效果对比]")
        print(f"  Precision: {baseline_metrics['precision']:.4f} -> {aug_metrics['precision']:.4f}")
        print(f"  Recall   : {baseline_metrics['recall']:.4f} -> {aug_metrics['recall']:.4f}")
        print(f"  F1-Score : {baseline_metrics['f1']:.4f} -> {aug_metrics['f1']:.4f}")
        
        # 保存最终模型
        final_model.save_model(config['model_path'])
        print(f"\n[保存] 最终模型已保存至: {config['model_path']}")
    else:
        # 不启用扩充，直接保存基线
        baseline_model.save_model(config['model_path'])
        final_model = baseline_model
        print(f"\n[保存] 基线模型已保存至: {config['model_path']}")
    
    # ---- 7.6 独立测试集验证 ----
    print("\n" + "="*60)
    print("Phase 3: 独立测试集最终验证")
    print("="*60)
    
    if os.path.exists(config['test_data_path']):
        X_test, y_test, _ = load_and_preprocess(
            config['test_data_path'], 
            scaler=scaler,  # 使用训练集的标准化器
            fit_scaler=False
        )
        evaluate_model(final_model, X_test, y_test, "独立测试集")
    else:
        print(f"[跳过] 未找到独立测试集: {config['test_data_path']}")


if __name__ == "__main__":
    main()
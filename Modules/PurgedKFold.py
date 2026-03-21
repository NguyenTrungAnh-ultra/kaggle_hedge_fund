import numpy as np
from sklearn.model_selection import KFold

class LeakFreePurgedKFold:
    def __init__(self, n_splits=5, pctEmbargo=0.01):
        self.n_splits = n_splits
        self.pctEmbargo = pctEmbargo
        
    def split(self, df):
        """
        df must contain 'ts_index' and 't1_end'
        """
        unique_ts = np.sort(df['ts_index'].unique())
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        
        embargo = int(len(unique_ts) * self.pctEmbargo)
        
        # Use underlying ndarrays for fast masking
        ts_index_arr = df['ts_index'].values
        t1_arr = df['t1_end'].values
        
        for train_ts_idx, test_ts_idx in kf.split(unique_ts):
            train_times = unique_ts[train_ts_idx]
            test_times = unique_ts[test_ts_idx]
            
            t0_test = test_times[0]
            t1_test = df[df['ts_index'].isin(test_times)]['t1_end'].max()
            
            # Initial boundaries
            base_train_mask = np.isin(ts_index_arr, train_times)
            test_indices = np.where(np.isin(ts_index_arr, test_times))[0]
            
            # 1. Purging (Loại bỏ rò rỉ phía trước và bao trùm)
            # Giữ lại nếu nhãn train kết thúc TRƯỚC test bắt đầu
            # HOẶC mẫu train bắt đầu SAU test kết thúc
            keep_before = t1_arr <= t0_test
            keep_after = ts_index_arr > t1_test
            
            # 2. Embargoing (Cấm vận phía sau)
            # Cấm các thời điểm train ngay sau t1_test
            if embargo > 0:
                after_test_times = unique_ts[unique_ts > t1_test]
                if len(after_test_times) > 0:
                    embargoed_times = after_test_times[:embargo]
                    # Loại bỏ các thời điểm trong embargo
                    keep_after = keep_after & (~np.isin(ts_index_arr, embargoed_times))
                    
            final_train_mask = base_train_mask & (keep_before | keep_after)
            train_indices = np.where(final_train_mask)[0]
            
            yield train_indices, test_indices
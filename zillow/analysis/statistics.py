import warnings
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from functools import cache

from scipy.stats import (
    ks_2samp, ttest_ind, mannwhitneyu, permutation_test, chi2_contingency,
    pearsonr, spearmanr, kendalltau, wasserstein_distance
)
from scipy.stats.contingency import association, odds_ratio
from scipy.spatial.distance import jensenshannon
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from dcor import distance_correlation

from ..utils.common import find_shared_cols, get_obj_features


# Many tests are calculated, but none of them is verified to follow 
# certain assumptions, even the dtype of data is chosen roughly. 
# The idea is to allow flexibility in calculation with combining
# an ability of a user to understand the type of data on their own. 
# Every calculation is based on common features. That means if you want
# to compare some features you have to set identical column names for them.
# This way it's possible to try any pair. 
# Probably the encapsulated type of features used per test is a bad code design. 

class SeparateDfFeatures:
    def __init__(self, df) -> None:
        # Not using sets to preserve the order of columns.
        self.categorical_features = df.select_dtypes(['category']).columns.tolist()
        self.boolean_features = df.select_dtypes(['bool']).columns.tolist()
        self.object_features = get_obj_features(df)
        self.continuous_features = [c for c in df.columns if 
                            c not in self.categorical_features and c not in self.boolean_features]
        self.numeric_features = [c for c in df.columns if 
                            c not in self.object_features and c not in self.boolean_features]

class CalcMethods:
    def __init__(self) -> None:
        self.list_of_methods = []
        self.common_features = []

    def calc_methods(self, methods: list[str] = 'all'):
        results = {}
        if methods == 'all':
            methods = self.list_of_methods

        for method in tqdm(methods):
            if hasattr(self, f"calc_{method}"):
                try:
                    results[method] = getattr(self, f"calc_{method}")()
                    print(f'Calculated {method}. ')
                except Exception as e:
                    print(f"Error occurred while calculating {method}: {e}.")

        return self._create_full_res_df(results)
    
    def _create_full_res_df(self, results: dict[str, pd.DataFrame]):
        res_df = pd.DataFrame(index=self.common_features)

        for _, test_res_df in results.items():
            result_indexed = test_res_df.set_index('feature')

            for col in result_indexed.columns:
                res_df[col] = result_indexed[col]

        return res_df

class GroupsTests:
    def __init__(self, df1, *args) -> None:
        self.common_features = find_shared_cols(df1.columns, *[df.columns for df in args])
        self.dfs = [df1[self.common_features]] + [df[self.common_features] for df in args]

class TwoSampTests:
    def __init__(self, df1, df2) -> None:
        self.common_features = df1.columns.intersection(df2.columns).tolist()
        self.df1 = df1[self.common_features]
        self.df2 = df2[self.common_features]

class StatTests(TwoSampTests, SeparateDfFeatures, CalcMethods):
    def __init__(self, df1, df2):
        TwoSampTests.__init__(self, df1, df2)
        SeparateDfFeatures.__init__(self, self.df1)

class DriftDetectionTests(StatTests):
    def __init__(self, df1, df2, bining_method='fd'):
        super().__init__(df1, df2)
        self.bining_method = bining_method
        self.avoid_zero_const = 1e-4
        self.list_of_methods = ['ks', 'psi', 'kl_div', 'jsd', 'wasserstein_dist']

    def _get_features_counts(self, expected: pd.Series, actual: pd.Series):
        # Categories of counts should be aligned.
        # Binning handles it, but value counts (cat col) doesn't.
        
        if expected.name in self.continuous_features:
            cnt0, cnt1 = self._bin_continuous_feats(expected, actual)
            n0, n1 = expected.count(), actual.count()
            if n0 == 0 or n1 == 0:
                warnings.warn(f"One of the series for feature {expected.name} is empty. Returning zero counts.")
                cnt0, cnt1 = np.array([0]), np.array([0])
            else:
                cnt0, cnt1 = cnt0 / n0, cnt1 / n1
        else:
            cnt0, cnt1 = expected.value_counts(normalize=True), actual.value_counts(normalize=True)
            cnt0, cnt1 = self._align_cnt_serieses(cnt0, cnt1)

        # To avoid math errors.
        return cnt0 + self.avoid_zero_const, cnt1 + self.avoid_zero_const

    def _bin_continuous_feats(self, expected:pd.Series, actual:pd.Series):
        # Remove outliers for correct bining. 
        expected = expected.dropna().astype(float)
        actual = actual.dropna().astype(float)
        edges = self._get_bin_edges(expected)

        counts_per_bin_expected = np.histogram(expected, bins=edges)[0]
        counts_per_bin_actual = np.histogram(actual, bins=edges)[0]
        return counts_per_bin_expected, counts_per_bin_actual

    def _get_bin_edges(self, array: pd.Series):
        array = array.clip(lower=array.quantile(0.005), upper=array.quantile(0.995)).copy()
        return np.histogram_bin_edges(array, bins=self.bining_method)
    
    def _align_cnt_serieses(self, cnt0, cnt1):
        all_indices = sorted(set(cnt0.index).union(set(cnt1.index)))
        cnt0_aligned = pd.Series([cnt0.get(i, self.avoid_zero_const) for i in all_indices], index=all_indices)
        cnt1_aligned = pd.Series([cnt1.get(i, self.avoid_zero_const) for i in all_indices], index=all_indices)

        # Without renormalization (assume self.avoid_zero_const is very small)
        return cnt0_aligned, cnt1_aligned

    def calc_ks(self):
        ks_results = []
        for feature in tqdm(self.continuous_features):
            ks_stat, p_value = ks_2samp(self.df1[feature], self.df2[feature])
            ks_results.append((feature, ks_stat, p_value))

        res_df = pd.DataFrame(ks_results, columns=['feature', 'ks_stat', 'ks_p_value'])
        return res_df
    
    def calc_psi(self):
        psi_results = []
        for feature in tqdm(self.common_features):
            cnt0, cnt1 = self._get_features_counts(self.df1[feature], self.df2[feature])
            psi = np.sum((cnt1 - cnt0) * np.log(cnt1 / cnt0))
            psi_results.append((feature, psi))

        res_df = pd.DataFrame(psi_results, columns=['feature', 'psi'])
        return res_df
    
    def calc_kl_div(self):
        kl_results = []
        for feature in tqdm(self.numeric_features):
            cnt0, cnt1 = self._get_features_counts(self.df1[feature], self.df2[feature])
            
            kl = np.sum(cnt1 * np.log(cnt1 / cnt0))
            kl_results.append((feature, kl))

        res_df = pd.DataFrame(kl_results, columns=['feature', 'kl_div'])
        return res_df
    
    def calc_jsd(self):
        jsd_results = []
        for feature in tqdm(self.numeric_features):
            cnt0, cnt1 = self._get_features_counts(self.df1[feature], self.df2[feature])
            jsd = jensenshannon(cnt0, cnt1) ** 2
            jsd_results.append((feature, jsd))

        res_df = pd.DataFrame(jsd_results, columns=['feature', 'jsd'])
        return res_df

    def calc_wasserstein_dist(self):
        wasserstein_results = []
        for feature in tqdm(self.numeric_features):
            dist = wasserstein_distance(self.df1[feature].dropna(), self.df2[feature].dropna())
            wasserstein_results.append((feature, dist))

        res_df = pd.DataFrame(wasserstein_results, columns=['feature', 'wasserstein_dist'])
        return res_df

class TwoSampHypothesisTests(StatTests):
    def __init__(self, df1, df2):
        super().__init__(df1, df2)
        self.list_of_methods = ['ttest', 'mannwhitney', 'ks', 'perms', 'chi2']

    def calc_ttest(self):
        ttest_results = []
        for feature in tqdm(self.numeric_features):
            t_stat, p_value = ttest_ind(self.df1[feature].dropna(), self.df2[feature].dropna())
            ttest_results.append((feature, t_stat, p_value))

        res_df = pd.DataFrame(ttest_results, columns=['feature', 't_stat', 't_p_value'])
        return res_df

    def calc_mannwhitney(self):
        mannwhitney_results = []
        for feature in tqdm(self.numeric_features):
            stat, p_value = mannwhitneyu(self.df1[feature].dropna(), self.df2[feature].dropna())
            mannwhitney_results.append((feature, stat, p_value))

        res_df = pd.DataFrame(mannwhitney_results, columns=['feature', 'mw_stat', 'mw_p_value'])
        return res_df
    
    def calc_ks(self):
        ks_results = []
        for feature in tqdm(self.numeric_features):
            ks_stat, p_value = ks_2samp(self.df1[feature].dropna(), self.df2[feature].dropna())
            ks_results.append((feature, ks_stat, p_value))

        res_df = pd.DataFrame(ks_results, columns=['feature', 'ks_stat', 'ks_p_value'])
        return res_df

    def calc_perms(self, statistic='default', n_resamples=9999):
        permutation_results = []
        if statistic == 'default':
            statistic = self._default_perms_statistic()
        for feature in tqdm(self.numeric_features):
            res = permutation_test(
                (self.df1[feature].dropna(), self.df2[feature].dropna()),
                statistic=statistic, n_resamples=n_resamples)
            permutation_results.append((feature, res.statistic, res.pvalue))

        res_df = pd.DataFrame(permutation_results, columns=['feature', 'perms_stat', 'perms_p_value'])
        return res_df
    
    def _default_perms_statistic(self):
        def mw_statistic(x, y):
            return mannwhitneyu(x, y).statistic
        return mw_statistic

    
    def calc_chi2(self):
        chi2_results = []
        for feature in tqdm(self.categorical_features):
            contingency_table = pd.crosstab(self.df1[feature], self.df2[feature])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi2_results.append((feature, chi2, p_value))

        res_df = pd.DataFrame(chi2_results, columns=['feature', 'chi2_stat', 'chi2_p_value'])
        return res_df

class CategoricalEffectSizes(StatTests):
    def __init__(self, df1, df2):
        TwoSampTests.__init__(self, df1, df2)
        SeparateDfFeatures.__init__(self, self.df1)
        self.list_of_methods = ['cramers_v', 'odds_ratio']

    def calc_cramers_v(self):
        cramers_v_results = []
        for feature in tqdm(self.categorical_features):
            contingency_table = pd.crosstab(self.df1[feature], self.df2[feature])
            cramers_v = association(contingency_table, method='cramer')
            cramers_v_results.append((feature, cramers_v))

        res_df = pd.DataFrame(cramers_v_results, columns=['feature', 'cramers_v'])
        return res_df
    
    def calc_odds_ratio(self):
        odds_ratio_results = []
        for feature in tqdm(self.boolean_features):
            contingency_table = pd.crosstab(self.df1[feature], self.df2[feature])
            or_stat = odds_ratio(contingency_table)
            odds_ratio_results.append((feature, or_stat))

        res_df = pd.DataFrame(odds_ratio_results, columns=['feature', 'odds_ratio'])
        return res_df

class TtestBasedEffectSizes(StatTests):
    def __init__(self, df1, df2):
        super().__init__(df1, df2)
        self.list_of_methods = ['cohen_d', 'hedges_g', 'glass_d']

    def _calc_cohen_d_from_features(self, f0, f1):
        n0 = len(f0)
        n1 = len(f1)
        mean_diff = f0.mean() - f1.mean()
        pooled_std = np.sqrt((f0.var() * (n0 - 1) + f1.var() * (n1 - 1)) / (n0 + n1 - 2))
        return mean_diff / pooled_std
    
    def calc_cohen_d(self):
        cohen_d_results = []
        for feature in tqdm(self.continuous_features):
            d = self._calc_cohen_d_from_features(self.df1[feature], self.df2[feature])
            cohen_d_results.append((feature, d))

        res_df = pd.DataFrame(cohen_d_results, columns=['feature', 'cohen_d'])
        return res_df

    def calc_hedges_g(self):
        hedges_g_results = []
        for feature in tqdm(self.continuous_features):
            n1 = len(self.df1[feature])
            n2 = len(self.df2[feature])
            d = self._calc_cohen_d_from_features(self.df1[feature], self.df2[feature])
            g = d * (1 - 3 / (4 * (n1 + n2) - 9))
            hedges_g_results.append((feature, g))

        res_df = pd.DataFrame(hedges_g_results, columns=['feature', 'hedges_g'])
        return res_df

    def calc_glass_d(self):
        # Second dataframe is control. 
        glass_d_results = []
        for feature in tqdm(self.continuous_features):
            std_control = self.df1[feature].std()
            d = (self.df1[feature].mean() - self.df2[feature].mean()) / std_control
            glass_d_results.append((feature, d))

        res_df = pd.DataFrame(glass_d_results, columns=['feature', 'glass_d'])
        return res_df
        
class GroupBasedEffectSizes(GroupsTests, SeparateDfFeatures, CalcMethods):
    def __init__(self, df1, *args) -> None:
        GroupsTests.__init__(self, df1, *args)
        SeparateDfFeatures.__init__(self, self.dfs[0])
        self.list_of_methods = ['eta2', 'omega2']
        
        self.combined_dfs_with_group = pd.concat(self.dfs, axis=0)
        self.combined_dfs_with_group['group'] = pd.Series(np.concatenate([[i] * len(df) 
            for i, df in enumerate(self.dfs)]), index=self.combined_dfs_with_group.index)
        
    @cache
    def _get_anova_table(self, feature):
        cur_comb_dfs = self.combined_dfs_with_group[[feature, 'group']].dropna()
        model = ols(f"{feature} ~ C(group)", data=cur_comb_dfs).fit()
        anova_table = anova_lm(model, typ=2)
        return anova_table

    def calc_eta2(self):
        eta2_results = []
        for feature in tqdm(self.continuous_features):
            anova_table = self._get_anova_table(feature)
            ss_between = anova_table['sum_sq']['C(group)']
            ss_total = anova_table['sum_sq'].sum()
            eta2 = ss_between / ss_total
            eta2_results.append((feature, eta2))

        res_df = pd.DataFrame(eta2_results, columns=['feature', 'eta2'])
        return res_df

    def calc_omega2(self):
        omega2_results = []
        for feature in tqdm(self.continuous_features):
            omega2 = self._calc_omega2_from_anova_table(
                self._get_anova_table(feature)
            )
            omega2_results.append((feature, omega2))

        res_df = pd.DataFrame(omega2_results, columns=['feature', 'omega2'])
        return res_df

    def _calc_omega2_from_anova_table(self, anova_table):
        ms_within = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
        ss_between = anova_table.loc['C(group)', 'sum_sq']
        df_between = anova_table.loc['C(group)', 'df']
        ss_total = anova_table['sum_sq'].sum()

        omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within)
        return max(0, omega2) # Omega is nonnegative
    
class TwoSampRelationshipEffectSizes(StatTests):
    def __init__(self, df1, df2, random_state=42):
        super().__init__(df1, df2)
        self.list_of_methods = ['pearson', 'distance_correlation', 'spearman', 'kendall', 
                                'point_biserial', 'rank_biserial', 'phi_coef']
        self.random_state = random_state
        self.discrete_features = find_shared_cols(self.numeric_features, self.categorical_features)
        
        # Note: Spearman and kendall are calculated for numeric features, including
        # continous, nominal and finally ordinal features. 
        # Note: Second df is assumed to be boolean, discrete, etc. 
        # (for point biserial and other tests). 

    def _prep_arrays(self, x, y):
        # For most tests the inputs must have same length. 
        min_sample = min(x.count(), y.count())
        return (x.dropna().sample(n=min_sample, random_state=self.random_state), 
                y.dropna().sample(n=min_sample, random_state=self.random_state))

    def calc_pearson(self):
        pearson_results = []
        for feature in tqdm(self.continuous_features):
            corr, p_value = pearsonr(*self._prep_arrays(self.df1[feature], self.df2[feature]))
            pearson_results.append((feature, corr, p_value))

        res_df = pd.DataFrame(pearson_results, columns=['feature', 'pearson_corr', 'pearson_p_value'])
        return res_df

    def calc_distance_correlation(self):
        distance_results = []
        for feature in tqdm(self.continuous_features):
            x, y = self._prep_arrays(self.df1[feature], self.df2[feature])
            # Has to be exactly float.
            x = x.astype(float, errors='raise')
            y = y.astype(float, errors='raise')
            corr = distance_correlation(x, y)
            distance_results.append((feature, corr))

        res_df = pd.DataFrame(distance_results, columns=['feature', 'distance_corr'])
        return res_df

    def calc_spearman(self):
        spearman_results = []
        for feature in tqdm(self.numeric_features):
            corr, p_value = spearmanr(*self._prep_arrays(self.df1[feature], self.df2[feature]))
            spearman_results.append((feature, corr, p_value))

        res_df = pd.DataFrame(spearman_results, columns=['feature', 'spearman_corr', 'spearman_p_value'])
        return res_df

    def calc_kendall(self):
        kendall_results = []
        for feature in tqdm(self.numeric_features):
            corr, p_value = kendalltau(*self._prep_arrays(self.df1[feature], self.df2[feature]))
            kendall_results.append((feature, corr, p_value))

        res_df = pd.DataFrame(kendall_results, columns=['feature', 'kendall_corr', 'kendall_p_value'])
        return res_df
    
    def calc_point_biserial(self):
        # Essentially the same with pearson.
        # Still pearson is calculated for both continous features.
        # Thus 'bool' dtype should be removed (replaced to numeric (e.g. float)).
        return self.calc_pearson()
    
    def calc_rank_biserial(self):
        rank_biserial_results = []
        for feature in tqdm(self.discrete_features):
            rank_biserial = self._calc_rank_biserial_formula(
                self.df1[feature].dropna(), self.df2[feature].dropna()
            )
            rank_biserial_results.append((feature, rank_biserial))

        res_df = pd.DataFrame(rank_biserial_results, columns=['feature', 'rank_biserial'])
        return res_df

    def _calc_rank_biserial_formula(self, x, y):
        group0 = x[y]
        group1 = x[~y]
        combined = pd.concat([group0, group1])
        ranks = combined.rank()
        r1 = ranks[:len(group0)].mean()
        r2 = ranks[len(group0):].mean()
        return 2 * (r1 - r2) / len(combined)

    def calc_phi_coef(self):
        phi_results = []
        for feature in tqdm(self.boolean_features):
            contingency_table = pd.crosstab(self.df1[feature], self.df2[feature])
            phi = self._calc_phi_formula(contingency_table)
            phi_results.append((feature, phi))

        res_df = pd.DataFrame(phi_results, columns=['feature', 'phi_coef'])
        return res_df

    def _calc_phi_formula(self, contingency_table):
        a, b, c, d = contingency_table.values.flatten()
        return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

class TwoSampNonParamEffectSizes(StatTests):
    def __init__(self, df1, df2):
        super().__init__(df1, df2)
        self.list_of_methods = ['cliff_delta', 'common_language']

    def calc_cliff_delta(self):
        cliff_delta_results = []
        for feature in tqdm(self.continuous_features):
            cliff_delta = self._calc_cliff_delta_formula(
                self.df1[feature].dropna(), self.df2[feature].dropna()
            )
            cliff_delta_results.append((feature, cliff_delta))

        res_df = pd.DataFrame(cliff_delta_results, columns=['feature', 'cliff_delta'])
        return res_df

    def _calc_cliff_delta_formula(self, x, y):
        more = np.sum(x[:, None] > y)
        less = np.sum(x[:, None] < y)
        return (more - less) / (len(x) * len(y))
    
    def calc_common_language(self):
        common_language_results = []
        for feature in tqdm(self.continuous_features):
            common_language = self._calc_common_language_formula(
                self.df1[feature].dropna(), self.df2[feature].dropna()
            )
            common_language_results.append((feature, common_language))

        res_df = pd.DataFrame(common_language_results, columns=['feature', 'common_language'])
        return res_df

    def _calc_common_language_formula(self, x, y):
        total = np.sum(x[:, None] > y) + 0.5 * np.sum(x[:, None] == y)
        return total / (len(x) * len(y))

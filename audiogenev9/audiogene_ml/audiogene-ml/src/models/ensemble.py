from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted
from sklearn_pandas import DataFrameMapper, gen_features
from typing import Iterable

from src.constants import AUDIOGRAM_LABELS


class PartitionedVotingClassifier(VotingClassifier):
    # Child class to the sklearn.ensemble.VotingClassifier
    #   The native sklearn-VC does cannot handle sub-models that predict different numbers of labels
    #   Be careful with future versions of sklearn. This is changing a fairly fundamental part of the class
    #    and future versions may make changes incompatible with the behavior of this class.
    def __init__(self,
                 estimators,
                 *,
                 voting="hard",
                 weights=None,
                 n_jobs=None,
                 flatten_transform=True,
                 verbose=False):
        if not isinstance(estimators[0][1], Pipeline):
            estimators = _PartitionedEnsembleHelper.construct_ensemble(estimators)
        super().__init__(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose
        )

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. Adjust results size by adding
        a column of zeros for classes the clf does not see in its training data."""
        probs = []
        for clf in self.estimators_:
            out = clf.predict_proba(X)
            missing_classes = np.setdiff1d(self.le_.transform(self.classes_), clf.classes_)
            if len(missing_classes) != 0:
                for c in missing_classes: out = np.insert(out, c, 0, axis=1)
            probs.append(out)
        return np.asarray(probs)


class PartitionedStackingClassifier(StackingClassifier):
    # Child class to the sklearn.ensemble.StackingClassifier
    #   The native sklearn-SC does cannot handle sub-models that predict different numbers of labels
    #   Be careful with future versions of sklearn. This is changing a fairly fundamental part of the class
    #    and future versions may make changes incompatible with the behavior of this class.
    def __init__(self,
                 estimators,
                 final_estimator=None,
                 *,
                 cv=None,
                 stack_method="auto",
                 n_jobs=None,
                 passthrough=False,
                 verbose=0
                 ):
        if not isinstance(estimators[0][1], Pipeline):
            estimators = _PartitionedEnsembleHelper.construct_ensemble(estimators)
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose
        )
        self.le_ = None

    def fit(self, X, y, sample_weights=None):
        super().fit(X, y, sample_weights)
        self.le_ = self._le

    def _transform(self, X):
        check_is_fitted(self)
        predictions = []
        for est, meth in zip(self.estimators_, self.stack_method_):
            if est != "drop":
                out = getattr(est, meth)(X)
                missing_classes = np.setdiff1d(self._le.transform(self.classes_), est.classes_)
                if len(missing_classes) != 0:
                    for c in missing_classes:
                        out = np.insert(out, c, 0, axis=1)
                predictions.append(out)
        return self._concatenate_predictions(X, predictions)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    # Credit to: https://medium.com/@literallywords/sklearn-identity-transformer-fcc18bac0e98
    def __init__(self): pass

    def fit(self, X, y=None): return self

    def transform(self, X, y=None): return X


class AudiogramFeatureFilters:
    @staticmethod
    def filter_ages(X, y, age_group, col_num=1):
        col = X[:, col_num]
        if '-' in age_group:
            ages = age_group.split('-')
            retained = (col >= float(ages[0])) & (col < float(ages[1]))
        elif '+' in age_group:
            ages = age_group.split('+')
            retained = col >= float(ages[0])
        else:
            retained = np.arange(X.shape[0])
        return X[retained], y[retained]

    @staticmethod
    def filter_shapes(X, y, shape, age_group, col_num=-2):
        _X, _y = AudiogramFeatureFilters.filter_ages(X, y, age_group)
        retained = _X[:, col_num] == shape
        return _X[retained], _y[retained]

    @staticmethod
    def filter_instance_groups(X, y, instance_group, col_num=-1):
        retained = X[:, col_num] == instance_group
        return X[retained], y[retained]

    @staticmethod
    def pass_filter(X, y):
        return X, y

    @staticmethod
    def filter_by_class_size(X, y, lower, upper):
        """
        Select samples from genes with a number of samples between the bounds detonated by thresholds
        :param X: Features Dataframe (n_samples, n_features)
        :param y: Class Series (n_samples,)
        :param lower: Minimum number of instances for inclusion in the class size group
        :param upper: Maximum number of instances for inclusion in the class size group
        :return: _X (n_samples_in_group, n_features), _y (n_samples_in_group,), cls_counts (n_classes_retained,)
        """
        if lower >= upper:
            raise ValueError(
                f"Expected bounds lower and upper s.t. lower < upper. Got lower bound {lower} >= upper bound {upper}.")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        cls_counts = pd.Series.value_counts(y)

        classes_in_range = cls_counts.loc[(lower < cls_counts) & (cls_counts < upper)]
        retained = y.isin(classes_in_range.index)
        return X[retained], y[retained], classes_in_range


class _PartitionedEnsembleHelper:
    """
    Contains most of the code specific to constructing AudioGene 9
    """
    feature_def = gen_features(columns=AUDIOGRAM_LABELS, classes=[IdentityTransformer])
    col_drop_mapper = DataFrameMapper(feature_def, input_df=True, df_out=True)

    @staticmethod
    def construct_ensemble(models):
        resampled_submodel_pipelines = []
        group_names = ('large', 'medium', 'small')
        size_group_thresholds = ((200, 10E10), (20, 200), (0, 20))
        resampling_targets = (None, 40, 10)
        directions = ('down', 'up', 'up')

        all_resampling_criteria = dict()

        for name, tgt, thresholds, dir_ in zip(group_names, resampling_targets, size_group_thresholds, directions):
            all_resampling_criteria[name] = {
                'resampling_target': tgt,
                'size_thresholds': thresholds,
                'direction': dir_
            }
        # Each pipeline must
        # Filter the group
        # Drop the non-audiogram columns (age_group, instance_group, shape)

        for model_type, alg, kwargs in models:
            resampler = None
            if model_type in ('large', 'medium', 'small'):
                size = model_type
                filter_ = FunctionSampler(func=AudiogramFeatureFilters.filter_instance_groups,
                                          kw_args={'instance_group': size})
                criteria = all_resampling_criteria[size]
                resampler = FunctionSampler(func=_PartitionedEnsembleHelper.resampling, kw_args={
                    'size_thresholds': criteria['size_thresholds'],
                    'sample_to': criteria['resampling_target'],
                    'direction': criteria['direction'],
                    'random_state': 30
                })
            elif model_type in ('0-20', '20+'):
                age = model_type
                filter_ = FunctionSampler(func=AudiogramFeatureFilters.filter_ages, kw_args={'age_group': age})
            elif model_type == 'base':
                filter_ = FunctionSampler(func=AudiogramFeatureFilters.pass_filter)
            else:
                shape, age = model_type.split('_')
                filter_ = FunctionSampler(func=AudiogramFeatureFilters.filter_shapes,
                                          kw_args={'shape': shape, 'age_group': age})

            # Voting classifier takes a list of tuples with the first object as a str:
            # name and the second as the transformer/pipeline/estimator
            steps = [(f"partition filter", filter_),
                     ('extract audiogram', _PartitionedEnsembleHelper.col_drop_mapper),
                     ('algorithm', alg() if kwargs is None else alg(**kwargs))]
            if resampler:
                # Instance resampling must occur before the non-audiogram columns are dropped
                steps.insert(1, ('instance resampler', resampler))
            resampled_submodel_pipelines.append(
                (f"{model_type}-pipe",
                 # Submodel pipeline
                 Pipeline(steps)
                 )
            )
        return resampled_submodel_pipelines

    @staticmethod
    def resampling(X, y, size_thresholds: Iterable[int], direction: str, sample_to: int = None, random_state: int = 30):
        # Argument validation
        if direction not in ('up', 'down', 'both', 'neither'):
            raise ValueError(f"Direction of resampling but be one of 'up', 'down', or 'both'. Got {direction}")
        try:
            low, high = size_thresholds
        except ValueError:
            raise ValueError(
                f"Expected 2 values, lower bound and upper bound, for size_thresholds. Got {len(size_thresholds)}")

        incompatible_msg = f"Incompatible resampling parameters. Cannot {direction}-sample to {sample_to} within range {low}-{high}"
        if direction in ('up', 'both') and sample_to < low:
            raise ValueError(incompatible_msg)
        elif direction == ('down', 'both') and sample_to > high:
            raise ValueError(incompatible_msg)

        if 'direction' in ('up', 'both') and sample_to > high:
            raise RuntimeWarning(
                f"Illogical to up-sample to {sample_to} when the class count range upper bound is {high}.")
        if 'direction' in ('down', 'both') and sample_to < low:
            raise RuntimeWarning(
                f"Illogical to down-sample to {sample_to} when the class count range lower bound is {low}.")
        # End argument validation

        # Filter down to only appropriate classes
        f_X, f_y, cls_counts = AudiogramFeatureFilters.filter_by_class_size(X,
                                                                            y,
                                                                            low,
                                                                            high)

        if not sample_to:
            sample_to = cls_counts.iloc[-1]
        resampled = []
        for cls, n in cls_counts.items():
            # Only resample with replacement if the specific class needs up-sampling
            # replacement = direction in ('up', 'both') and n < sample_to
            replacement = True
            in_cls = f_y == cls
            _X = f_X[in_cls]
            _y = f_y[in_cls]
            resampled.append(
                resample(_X, _y, replace=replacement, n_samples=sample_to, random_state=random_state, stratify=None))

        r_X, r_y = np.vstack([s[0] for s in resampled]), np.concatenate([np.array(s[1]) for s in resampled])
        return r_X, r_y


import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

from .utils.distance import distance, pairwise_distance
from .utils.mean import gmean
from .utils.utils import check_metric
from .classification import MDM


def multiset_perm_number(y):
    """return the number of unique permutation in a multiset."""
    pass


def unique_permutations(elements):
    """Return the list of unique permutations."""
    pass


class BasePermutation():
    """Base object for permutations test"""

    def test(self, X, y, groups=None, verbose=True):
        """Performs the permutation test

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array at
            least 2d.
        y : array-like
            The target variable to try to predict in the case of
            supervised learning.
        groups : array-like, default=None
            Group labels used while splitting the dataset into train/test set.
        verbose : bool, default=True
            If true, print progress.
        """
        pass

    def _print_progress(self, ii):
        """Print permutation progress"""
        pass

    def _initial_transform(self, X):
        """Initial transformation. By default return X."""
        pass

    def _shuffle(self, y, groups, rs):
        """Return a shuffled copy of y eventually shuffle among same groups."""
        pass

    def plot(self, nbins=10, range=None, axes=None):
        """Plot results of the permutation test.

        Parameters
        ----------
        nbins : integer or array-like or "auto", default=10
            If an integer is given, bins + 1 bin edges are returned,
            consistently with np.histogram() for numpy version >= 1.3.
            Unequally spaced bins are supported if bins is a sequence.
        range : tuple or None, default=None
            The lower and upper range of the bins. Lower and upper outliers are
            ignored. If not provided, range is (x.min(), x.max()).
            Range has no effect if bins is a sequence.
            If bins is a sequence or range is specified, autoscaling is based
            on the specified bin range instead of the range of x.
        axes : axes handle, default=None
            Axes handle for matplotlib. if None a new figure will be created.
        """
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        axes.hist(self.scores_[1:], nbins, range=range)
        x_val = self.scores_[0]
        y_max = axes.get_ylim()[1]
        axes.plot([x_val, x_val], [0, y_max], "--r", lw=2)
        x_max = axes.get_xlim()[1]
        x_min = axes.get_xlim()[0]
        x_pos = x_min + ((x_max - x_min) * 0.25)
        axes.text(x_pos, y_max * 0.8, "p-value: %.3f" % self.p_value_)
        axes.set_xlabel("Score")
        axes.set_ylabel("Count")
        return axes


class PermutationModel(BasePermutation):
    """Permutation test using any scikit-learn model for scoring.

    Perform a permutation test using the cross-validation score of any
    scikit-learn compatible model. Score is obtained with ``cross_val_score``
    from scikit-learn. The score should be a "higer is better" metric.

    Parameters
    ----------
    n_perms : int, default=100
        The number of permutation. The minimum should be 20 for a resolution of
        0.05 p-value.
    model : sklearn compatible model, default=MDM()
        The model for scoring.
    cv : int or cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation;
        - integer, to specify the number of folds in a ``(Stratified)KFold``;
        - an object to be used as a cross-validation generator;
        - an iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, ``StratifiedKFold`` is used. In all
        other cases, ``KFold`` is used.
    scoring : string or callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    n_jobs : integer, default=1
        The number of CPUs to use to do the computation. -1 means "all CPUs".
    random_state : int, default=42
        random state for the permutation test.

    Attributes
    ----------
    p_value_ : float
        the p-value of the test
    scores_ : list
        contain all scores for all permutations. The fist element is the
        non-permuted score.

    See Also
    --------
    PermutationDistance
    """

    def __init__(
        self,
        n_perms=100,
        model=MDM(),
        cv=3,
        scoring=None,
        n_jobs=1,
        random_state=42,
    ):
        """Init."""
        self.n_perms = n_perms
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

    def score(self, X, y, groups=None):
        """Score one permutation.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array at
            least 2d.
        y : array-like
            The target variable to try to predict in the case of
            supervised learning.
        groups : array-like, default=None
            Group labels used while splitting the dataset into train/test set
        """
        pass


class PermutationDistance(BasePermutation):
    """Permutation test based on distance.

    Perform a permutation test based on distance. You have the choice of 3
    different statistics:

    - "pairwise":
        the statistic is based on paiwire distance as
        described in [1]_. This is the fastest option for low sample size since
        the pairwise distance matrix does not need to be estimated for each
        permutation.

    - "ttest":
        t-test based statistic obtained by the ration of the
        distance between each centroid and the group dispersion.
        The means have to be estimated for each permutation, leading to a
        slower procedure. However, this can be used for high sample size.

    - "ftest":
        f-test based statistic estimated using the between and
        within group variability. As for the "ttest" stats, group centroid
        are estimated for each permutation.

    Parameters
    ----------
    n_perms : int, default=100
        The number of permutation. The minimum should be 20 for a resolution of
        0.05 p-value.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.gmean`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    mode : string, default="pairwise"
        Type of statistic to use. could be "pairwise", "ttest" of "ftest"
    n_jobs : integer, default=1
        The number of CPUs to use to do the computation. -1 means "all CPUs".
    random_state : int, default=42
        random state for the permutation test.
    estimator : None or sklearn compatible estimator, default=None
        If provided, data are transformed before every permutation. should
        not be used unless a supervised opperation must be applied on the data.
        This would be the case for ERP covariance.

    Attributes
    ----------
    p_value_ : float
        the p-value of the test
    scores_ : list
        contain all scores for all permutations. The fist element is the
        non-permuted score.

    See Also
    --------
    PermutationModel

    References
    ----------
    .. [1] `A new method for non-parametric multivariate analysis of variance
        <https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x>`_
        M. Anderson. Austral ecology, Volume 26, Issue 1, February 2001.
    """

    def __init__(
        self,
        n_perms=100,
        metric="riemann",
        mode="pairwise",
        n_jobs=1,
        random_state=42,
        estimator=None,
    ):
        """Init."""
        self.n_perms = n_perms
        if mode not in ["pairwise", "ttest", "ftest"]:
            raise ValueError("mode must be 'pairwise', 'ttest' or 'ftest'")
        self.mode = mode
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimator = estimator

    def score(self, X, y, groups=None):
        """Score of a permutation.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array at
            least 2d.
        y : array-like
            The target variable to try to predict in the case of
            supervised learning.
        groups : array-like, default=None
            Group labels used while splitting the dataset into train/test set
        """
        pass

    def _initial_transform(self, X):
        """Initial transform"""
        pass

    def __init_transform(self, X):
        """Init tr"""
        pass

    def _score_ftest(self, X, y):
        """Get the score"""
        pass

    def _score_ttest(self, X, y):
        """Get the score"""
        pass

    def _score_pairwise(self, X, y):
        """Score for the pairwise distance test."""
        pass

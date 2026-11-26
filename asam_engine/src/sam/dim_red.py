import sklearn.decomposition
from sam.coords import calc_dmap_triu


def run_pca(traj_ref, traj_hat, featurize_for_pca=None, get_x=False):
    # Function to featurize the trajectory and to get input features of PCA.
    if featurize_for_pca is None:
        featurize_for_pca = lambda x: calc_dmap_triu(x, backend="numpy")
    # PCA features.
    x_ref = featurize_for_pca(traj_ref.xyz)
    x_hat = featurize_for_pca(traj_hat.xyz)
    # Perform PCA.
    pca = sklearn.decomposition.PCA(n_components=10)
    pca.fit(x_ref)
    y_ref = pca.transform(x_ref)
    y_hat = pca.transform(x_hat)
    # Return the results.
    results = {"y_ref": y_ref, "y_hat": y_hat, "obj": pca}
    if get_x:
        results.update({"x_ref": x_ref, "x_hat": x_hat})
    return results
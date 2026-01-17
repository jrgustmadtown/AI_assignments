from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    x = x - np.mean(x, axis=0)

    return x

def get_covariance(dataset):
    # Your implementation goes here!
    x = dataset
    s = (1/(x.shape[0]-1)) * np.dot(np.transpose(x), x)

    return s

def get_eig(S, k):
    # Your implementation goes here!
    evals, evects = eigh(S, subset_by_index=[S.shape[0]-k, S.shape[0]-1], eigvals_only=False)
    sort = np.argsort(evals)[::-1]
    k_evals = evals[sort]
    k_evects = evects[:, sort]
    lamby = np.diag(k_evals) 
  
    return lamby, k_evects

def get_eig_prop(S, prop):
    # Your implementation goes here!
    evals, evects = eigh(S, eigvals_only=False)
    sort = np.argsort(evals)[::-1]
    evals = evals[sort]
    evects = evects[:, sort]
    vari = np.trace(S)
    exp_vari = evals / vari
    cumvari = np.cumsum(exp_vari)

    new_sort = np.searchsorted(cumvari, prop) + 1
    m_evals = evals[:new_sort]
    m_evects = evects[:, :new_sort]

    lamby = np.diagflat(m_evals)
   
    return lamby, m_evects

def project_and_reconstruct_image(image, U):
    # Your implementation goes here!
    a = np.dot(U.T, image)
    recon = np.dot(U, a)

    return recon

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    # Your implementation goes here!
    im_orig_fullres = im_orig_fullres.reshape(218, 178, 3)
    im_orig = im_orig.reshape(60,50)
    im_reconstructed = im_reconstructed.reshape(60,50)

    ax1.set_title("Original High Res")
    ax2.set_title("Original")
    ax3.set_title("Reconstructed")

    ax1.imshow(im_orig_fullres)
    orig = ax2.imshow(im_orig, cmap='gray', aspect='equal')
    recon = ax3.imshow(im_reconstructed, cmap='gray', aspect='equal')

    fig.colorbar(orig, ax=ax2)
    fig.colorbar(recon, ax=ax3)

    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    a = np.dot(U.T, image)
    gauss = np.random.normal(0, sigma, size=a.shape)
    a_p = a+gauss
    im_p = np.dot(U, a_p)

    return im_p

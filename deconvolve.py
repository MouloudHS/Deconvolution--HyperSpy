from skimage import measure

def get_psf_from_entropy(signal, p = 0.85):
    from skimage.measure import entropy
    
    entropy_map = signal.deepcopy()
    entropy_map.map(entropy)
    mean_entropy = entropy_map.data.mean()
    
    mean_entropy = entropy_map.data.mean()
    mask_entropy = entropy_map < p*mean_entropy
    s_low = signal*mask_entropy
    psf = s_low.sum()
    psf = psf/psf.data.max()

    return psf

def deconvolve(image, psf, itr = 50):
    from skimage.restoration import richardson_lucy
    
    norm_const = image.max()
    image = image/norm_const
    
    # Deconvolution
    image = richardson_lucy(image, psf, iterations=itr)
    
    # Multiply by the saved normalization constant
    image = image * norm_const

    return image

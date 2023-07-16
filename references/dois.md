# Misc general references

# Goodfellow et. al. deep learning book 
# Does not seem to have a doi ... will have to 
# generate bib entry manually. From the book's
# webpage the bibentry is:
# @book{Goodfellow-et-al-2016,
#    title={Deep Learning},
#    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
#    publisher={MIT Press},
#    note={\url{http://www.deeplearningbook.org}},
#    year={2016}
# }
#
# PDF available for FREE here:
# https://www.deeplearningbook.org/


# ML for computational fluid dynamics Steve Brunton YouTube video series
# https://www.youtube.com/playlist?list=PLMrJAkhIeNNQWO3ESiccZmPssvUDFHL4M

# Survey papers on ML techniques for computational fluid dynamics
doi:10.1038/s43588-022-00264-7
doi:10.1146/annurev-fluid-010719-060214

# Review paper on data science for geodynamics
# (PDF available in the repo)
doi:10.1016/B978-0-08-102908-4.00111-9

# An introduction to variational autoencoders (quite advanced)
doi:10.1561/2200000056

# Seminal paper in which the ML architecture of Convolutional 
# AutoEncoders (ConvAE) was proposed
doi:10.1007/978-3-642-21735-7_7

###################################################

# (some) State of the art references on ML-based approaches for Mantle convection

# ML-based FORWARD surrogate to predict 1D transient temperature profiles
# given a set of physical problem input parameters. The data set 
# used for training is built out of 1D transient temperature profiles
# extracted from a post-processing stage of 2D time evolving Mantle convection 
# simulations
doi:10.1093/gji/ggaa234 

# The extension of the previous work to 2D time-evolving Mantle convection 
# simulations. The ML-based FORWARD surrogate provides the full 2D temperature
# field evolution given the physical problem input parameters
doi:10.1103/PhysRevFluids.6.113801

# Just the previous reference in ArXiV to show that BibHandler.jl also supports
# ArXiV entries apart from DOIs
arxiv:2108.10105

# A fully probabilistic ML-based inversion method able to infer input model
# parameters out of observations of the mantle temperature field after billion
# of years of convection. This kind of ML-based system actually tackles the 
# inverse Mantle convection problem. 
# Note: there is also a PhD thesis form which this paper stems. I have added
# the PDF of the thesis to the git repo.
doi:10.1016/j.pepi.2016.05.016

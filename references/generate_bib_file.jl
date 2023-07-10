# Only required the first time we call the script.
# Anyway, it does not harm to have it here executed
# for any call to the script.

# MacOSx users: in some computers, we found ssh authentication errors 
# when the Julia package manager tries to talk
# to Github. If this is the case, a workaround is to clone the BibHandler.jl repo
# manually and use the instruction:
#
# Pkg.add(path="local_path_to_cloned_bibhandler_github_repo", rev="master")
#
# instead of the Pkg.add(...) instruction below

using Pkg
Pkg.add(url="git@github.com:BadiaLab/BibHandler.jl.git", rev="master")
Pkg.instantiate()

using BibHandler
file2bib("dois.md";download=false)

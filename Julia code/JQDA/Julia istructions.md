# Setup

1. Install Julia from the site https://julialang.org/downloads/
2. Open Julia and go in the JQDA folder using `cd("path/to/JQDA")`
3. Now you need to open the JQDA project. To do it type exactly:
	- `] activate .` (the ] bracket is to enter pkg mode)
	- `instantiate` (still in the pkg mode) this command is required only the first time, as it installs the missing packages
	- `canc` or `ctrl+c` to exit pkg mode
	- now you are done :)
4. Then to work on the code we use the easy Pluto editor, so in Julia type `using Pluto` and then `Pluto.run()`. This will open a page in your browser, where you can select the notebook on which we will work on.


# References
https://juliaimages.org/latest/
https://github.com/r3tex/ObjectDetector.jl
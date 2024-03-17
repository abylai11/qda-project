# Setup

1. Install Julia from the site https://julialang.org/downloads/
2. Open Julia and go in the JQDA folder using `cd("path/to/JQDA")`
3. Now you need to open the JQDA project. To do it type exactly:
	- `] activate .` (the ] bracket is to enter pkg mode)
	- `instantiate` (still in the pkg mode) this command is required only the first time, as it installs the missing packages
	- `canc` or `ctrl+c` to exit pkg mode
	- now you are done :)
4. Then to work on the code we use the easy Pluto editor, so in Julia type `using Pluto` and then `Pluto.run()`. This will open a page in your browser, where you can select the notebook on which we will work on.

After you did this you should get this output for the status (`st`) of the project

```julia
(@v1.9) pkg> activate .
  Activating project at `C:\Users\feder\Desktop\Uni magistrale\Quality data analysis\qda-project\Julia code\JQDA`

(JQDA) pkg> st
Status `C:\Users\feder\Desktop\Uni magistrale\Quality data analysis\qda-project\Julia code\JQDA\Project.toml`
  [5789e2e9] FileIO v1.16.2
  [4381153b] ImageDraw v0.2.6
  [92ff4b2b] ImageFeatures v0.5.2
  [80713f31] ImageSegmentation v1.8.2
  [02fcd773] ImageTransformations v0.10.1
  [916415d5] Images v0.26.0
  [3dfc1049] ObjectDetector v0.3.1
  [c3e4b0f8] Pluto v0.19.40
  [7f904dfe] PlutoUI v0.7.58
  [5e47fb64] TestImages v1.8.0
```

# References
https://juliaimages.org/latest/
https://github.com/r3tex/ObjectDetector.jl
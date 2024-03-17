### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ f2166155-6b17-46a6-bf2f-58476a16cb96
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 7855cb01-a95d-47de-8ef6-43b62af38db8
using PlutoUI

# ╔═╡ 21704e07-f4e4-436f-b10a-c50ed96dd886
using Images, TestImages

# ╔═╡ fbb5afcc-4fe9-47e5-8e36-d4e0895a1455
using FileIO

# ╔═╡ aba991a8-857e-49ea-bf20-f3968f1e46d0
using ObjectDetector

# ╔═╡ 0ca3370e-791c-46bb-8e07-383d254285d5
md"""
# Setup
"""

# ╔═╡ 54a14054-cd36-4d95-9a1a-993b874510ed
md"""
Dont touch here.\
Anyway something about Pluto:
- execute a cell with Shift+Enter
- create a new cell below with Ctrl+Enter
- each cell tracks the connections to subsequent cells, meaning that if we change something in a cell, all the cells that "depended" on that get also updated. Like if I change an image at the begninning, all the subsequent codes updates with the new image.
"""

# ╔═╡ 78269f2f-2e7d-44a5-a998-b57094722e88
pwd()

# ╔═╡ 0892ac49-4413-4b65-80c5-f40c0390dfd9
TableOfContents(title="📚 Table of Contents", indent=true, depth=3, aside=true)

# ╔═╡ a57c0c31-1891-4955-bd05-8d89ed578779
md"""
# Real code
"""

# ╔═╡ 18f71bf9-a511-4afe-a11d-027348acbd3d
rand_img = rand(RGB,20,20)

# ╔═╡ 31609e07-ef7e-44e3-b14a-06657ec0886e
tray = Gray.(load("../images/tray_example.png"))
# tray = testimage("cameraman")

# ╔═╡ b4b1cb67-c40b-4905-8a4a-a471a9d36f66
tray[100:400,100:400] # for example

# ╔═╡ 5219e0b4-d048-43dd-b752-a4b14824a985
md"""
## With NN
"""

# ╔═╡ 21bf3cb3-2eed-4b9c-904a-d91559736b3a
size(tray)

# ╔═╡ 52fc6b9a-6e13-4eb7-a256-353187eee650
img = imresize(tray, (608, 608))

# ╔═╡ 4c27e2ba-402e-4eff-b291-c7f08241daad
size(img)

# ╔═╡ d2d72dbf-59ba-4fd5-93db-2a3244a61bcb
md"""
Then from here we can start testing ImageSegmentation, Neural Networks, etc
"""

# ╔═╡ 79b93040-bc56-4db7-bde3-3a71558c81bd
# ╠═╡ show_logs = false
# yolomod = YOLO.v3_608_COCO(batch=1, silent=true)
# Load the YOLOv3-tiny model pretrained on COCO, with a batch size of 1
# other models
# yolomod = YOLO.v2_tiny_416_COCO(batch=1, silent=true)
yolomod = YOLO.v3_608_COCO(batch=1, silent=true)

# ╔═╡ bec7a2a6-ee5d-44b3-bbd9-1f819f973bdb
# ╠═╡ show_logs = false
batch = emptybatch(yolomod)
# Create a batch object. Automatically uses the GPU if available

# ╔═╡ 3e52ef58-e87c-418a-b875-a56a3c3f212f
# ╠═╡ show_logs = false
batch[:,:,:,1], padding = prepareImage(img, yolomod) 
# Send resized image to the batch

# ╔═╡ 8a1d2efa-efdb-4c33-ae57-8af9072b54f3
Float64.(img)

# ╔═╡ bb357471-7bb4-4968-822f-f8183ea36b12
res = yolomod(batch, detectThresh=0.1, overlapThresh=0.4)
# Run the model on the length-1 batch

# ╔═╡ 63d63187-1402-4ac8-881e-bdf7a334b3f1
imgBoxes = drawBoxes(img.+0.4, yolomod, padding, res)

# ╔═╡ 2f5e9001-e284-4965-8159-a00030a45e12
md"""
But it still unclear how it works, this was just a test and also a way to see (and show you) what Julia can do :)
"""

# ╔═╡ ec36238a-35fe-42c1-87f3-c9e088f5cc46
md"""
## With statistics
[https://juliaimages.org/latest/pkgs/segmentation/](https://juliaimages.org/latest/pkgs/segmentation/)
"""

# ╔═╡ ead3c71b-1027-4048-8833-7ab7eee4e90a
segments = felzenszwalb(tray, 2000, 30_000)
# first argument relates to how many labels are created
# second argument removes segments with fewer than 30000 pixels

# ╔═╡ 213607ff-95f2-46e3-a806-a22d9d5eed08
tray[170:370,120:320]

# ╔═╡ 30a8dab9-ab49-4902-8a08-27c1311ba5b8
(370-170)*(320-120)

# ╔═╡ d743a9f3-61e8-4d12-9496-1905e58699e0
# map(i->segment_mean(segments,i), labels_map(segments))
Gray.(labels_map(segments)./maximum(labels_map(segments)))

# ╔═╡ 64e7649a-a7ff-4603-9a35-67deb350e363
unique(labels_map(segments))

# ╔═╡ Cell order:
# ╟─0ca3370e-791c-46bb-8e07-383d254285d5
# ╟─54a14054-cd36-4d95-9a1a-993b874510ed
# ╠═f2166155-6b17-46a6-bf2f-58476a16cb96
# ╠═78269f2f-2e7d-44a5-a998-b57094722e88
# ╠═7855cb01-a95d-47de-8ef6-43b62af38db8
# ╠═0892ac49-4413-4b65-80c5-f40c0390dfd9
# ╠═a57c0c31-1891-4955-bd05-8d89ed578779
# ╠═21704e07-f4e4-436f-b10a-c50ed96dd886
# ╠═fbb5afcc-4fe9-47e5-8e36-d4e0895a1455
# ╠═18f71bf9-a511-4afe-a11d-027348acbd3d
# ╠═31609e07-ef7e-44e3-b14a-06657ec0886e
# ╠═b4b1cb67-c40b-4905-8a4a-a471a9d36f66
# ╟─5219e0b4-d048-43dd-b752-a4b14824a985
# ╠═21bf3cb3-2eed-4b9c-904a-d91559736b3a
# ╠═52fc6b9a-6e13-4eb7-a256-353187eee650
# ╠═4c27e2ba-402e-4eff-b291-c7f08241daad
# ╟─d2d72dbf-59ba-4fd5-93db-2a3244a61bcb
# ╠═aba991a8-857e-49ea-bf20-f3968f1e46d0
# ╠═79b93040-bc56-4db7-bde3-3a71558c81bd
# ╠═bec7a2a6-ee5d-44b3-bbd9-1f819f973bdb
# ╠═3e52ef58-e87c-418a-b875-a56a3c3f212f
# ╠═8a1d2efa-efdb-4c33-ae57-8af9072b54f3
# ╠═bb357471-7bb4-4968-822f-f8183ea36b12
# ╠═63d63187-1402-4ac8-881e-bdf7a334b3f1
# ╟─2f5e9001-e284-4965-8159-a00030a45e12
# ╟─ec36238a-35fe-42c1-87f3-c9e088f5cc46
# ╠═ead3c71b-1027-4048-8833-7ab7eee4e90a
# ╠═213607ff-95f2-46e3-a806-a22d9d5eed08
# ╠═30a8dab9-ab49-4902-8a08-27c1311ba5b8
# ╠═d743a9f3-61e8-4d12-9496-1905e58699e0
# ╠═64e7649a-a7ff-4603-9a35-67deb350e363

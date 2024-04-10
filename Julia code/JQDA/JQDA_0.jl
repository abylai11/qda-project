### A Pluto.jl notebook ###
# v0.19.39

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
using Images, TestImages, Random, ImageSegmentation

# ╔═╡ 330fbf6e-fa52-4c0e-96bc-9a8994f57a09
using ImageTransformations, CoordinateTransformations, Rotations

# ╔═╡ fbb5afcc-4fe9-47e5-8e36-d4e0895a1455
using FileIO

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
# Segmentation
"""

# ╔═╡ 18f71bf9-a511-4afe-a11d-027348acbd3d
rand_img = rand(RGB,20,20)

# ╔═╡ 31609e07-ef7e-44e3-b14a-06657ec0886e
tray = Gray.(load("../images/tray_example.png"))
# tray = testimage("cameraman")

# ╔═╡ b4b1cb67-c40b-4905-8a4a-a471a9d36f66
# then images are just matrices of pixels
tray[200:300,200:400] # for example

# ╔═╡ 5219e0b4-d048-43dd-b752-a4b14824a985
md"""
## With NN
"""

# ╔═╡ 52fc6b9a-6e13-4eb7-a256-353187eee650
# ╠═╡ disabled = true
#=╠═╡
img = imresize(tray, (608, 608))
  ╠═╡ =#

# ╔═╡ d2d72dbf-59ba-4fd5-93db-2a3244a61bcb
md"""
Then from here we can start testing ImageSegmentation, Neural Networks, etc
"""

# ╔═╡ aba991a8-857e-49ea-bf20-f3968f1e46d0
# ╠═╡ disabled = true
#=╠═╡
using ObjectDetector
  ╠═╡ =#

# ╔═╡ 79b93040-bc56-4db7-bde3-3a71558c81bd
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
# yolomod = YOLO.v3_608_COCO(batch=1, silent=true)
# Load the YOLOv3-tiny model pretrained on COCO, with a batch size of 1
# other models
# yolomod = YOLO.v2_tiny_416_COCO(batch=1, silent=true)
yolomod = YOLO.v3_608_COCO(batch=1, silent=true)
  ╠═╡ =#

# ╔═╡ bec7a2a6-ee5d-44b3-bbd9-1f819f973bdb
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
batch = emptybatch(yolomod)
# Create a batch object. Automatically uses the GPU if available
  ╠═╡ =#

# ╔═╡ 3e52ef58-e87c-418a-b875-a56a3c3f212f
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
batch[:,:,:,1], padding = prepareImage(img, yolomod) 
# Send resized image to the batch
  ╠═╡ =#

# ╔═╡ 8a1d2efa-efdb-4c33-ae57-8af9072b54f3
# ╠═╡ disabled = true
#=╠═╡
Float64.(img)
  ╠═╡ =#

# ╔═╡ bb357471-7bb4-4968-822f-f8183ea36b12
# ╠═╡ disabled = true
#=╠═╡
res = yolomod(batch, detectThresh=0.1, overlapThresh=0.4)
# Run the model on the length-1 batch
  ╠═╡ =#

# ╔═╡ 63d63187-1402-4ac8-881e-bdf7a334b3f1
# ╠═╡ disabled = true
#=╠═╡
imgBoxes = drawBoxes(img.+0.4, yolomod, padding, res)
  ╠═╡ =#

# ╔═╡ 2f5e9001-e284-4965-8159-a00030a45e12
md"""
But it is still unclear how it works, and it doesnt seem to work good anyway.
"""

# ╔═╡ ec36238a-35fe-42c1-87f3-c9e088f5cc46
md"""
## With statistics
[https://juliaimages.org/latest/pkgs/segmentation/](https://juliaimages.org/latest/pkgs/segmentation/)
[https://juliaimages.org/latest/pkgs/segmentation/#Result-1]
(https://juliaimages.org/latest/pkgs/segmentation/#Result-1)
[https://marketsplash.com/tutorials/julia/julia-image-processing/]
(https://marketsplash.com/tutorials/julia/julia-image-processing/)
"""

# ╔═╡ 346081ac-1caa-40cc-84df-8885c4ce2aba
md"""
### Felzenszwalb
"""

# ╔═╡ ead3c71b-1027-4048-8833-7ab7eee4e90a
FELZ = felzenszwalb(tray, 2000, 30_000)
# first argument relates to how many labels are created
# second argument removes segments with fewer than 30000 pixels

# ╔═╡ 213607ff-95f2-46e3-a806-a22d9d5eed08
tray[170:370,120:320]

# ╔═╡ 30a8dab9-ab49-4902-8a08-27c1311ba5b8
# total number of pixels occupied by the interesting object
(370-170)*(320-120)

# ╔═╡ 64e7649a-a7ff-4603-9a35-67deb350e363
unique(labels_map(FELZ))

# ╔═╡ a1850f3a-b491-4f71-8642-bac933417339
labels_map(FELZ)

# ╔═╡ 689ad7ae-e9c4-4481-b634-ac8211df55eb
function get_random_color(seed)
    Random.seed!(seed)
    rand(RGB{N0f8})
end

# ╔═╡ f0e6afd1-3032-4df0-959f-f2a89348c7ef
map(i->get_random_color(i), labels_map(FELZ))
# Gray.(labels_map(segments)./maximum(labels_map(segments)))

# ╔═╡ b9adec8d-a9e6-4d24-a64f-a214ef8bab1c
md"""
### Unseeded region growing
"""

# ╔═╡ 7951c08e-e05f-44be-9a07-853f4c3836b0
URG = unseeded_region_growing(tray, 0.15) # here 0.05 is the threshold

# ╔═╡ a55dc561-1405-4e92-9bc8-50be622762a7
map(i->get_random_color(i), labels_map(URG)) # bad
# Gray.(labels_map(seg)/6)

# ╔═╡ 5665c78f-0f26-47a8-9cc7-d936fb410195
md"""
### Fast scanning
"""

# ╔═╡ 39d7ffac-00ee-4f85-8850-45e3d883c6cf
FASTS = fast_scanning(tray, 0.10)

# ╔═╡ 5cd48d20-a223-4599-9cbb-a9c2dad33336
FAST = prune_segments(FASTS, 
	i->(segment_pixel_count(FASTS,i)<2000), 
	(i,j)->(-segment_pixel_count(FASTS,j)))

# ╔═╡ f84f29ea-6642-4243-ad88-f4b6ac6e9809
map(i->get_random_color(i), labels_map(FAST)) # a bit less bad
# Gray.(labels_map(FAST)./maximum(labels_map(FAST)))

# ╔═╡ 57fbdc8a-86a2-4041-a6bb-659b539c3d2b
md"""
### Region splitting
"""

# ╔═╡ 0d4c8b48-6c77-49c1-b954-4776083b67b5
function homogeneous(img)
	min, max = extrema(img)
	max - min < 0.4
end

# ╔═╡ ddda3d0f-5efa-4794-886f-76f3ba7dfba6
REGSPL = region_splitting(tray, homogeneous)

# ╔═╡ ec4acc55-68f7-4b80-8101-df4fc6444f3c
map(i->get_random_color(i), labels_map(REGSPL)) # cute but bad
# Gray.(labels_map(REGSPL)./maximum(labels_map(REGSPL)))

# ╔═╡ 0e948dac-4c47-48ff-80e8-bcf5b0179992
md"""
## Segmented region extraction
"""

# ╔═╡ 9d934cfb-c26a-4493-b717-0aaf8295237d
function trova_indici_pattern(m, label)
	# first = findfirst(x -> x==label, m)
	# last = findlast(x -> x==label, m)
	# start_row, start_col = first[1], first[2]
	# end_row, end_col = last[1], last[2]
	
	start_row, start_col = 0, 0
	end_row, end_col = 0, 0

	for i in 1:size(m,1)
		if label in m[i,:]
			if start_row==0 start_row = i end
			end_row = i
		end
	end
	for j in 1:size(m,2)
		if label in m[:,j]
			if start_col==0 start_col= j end
			end_col = j
		end
	end

    return start_row, start_col, end_row, end_col
end

# ╔═╡ 7f7891a6-14cc-42b8-b724-a39883a2308b
# example
m = [
    1 1 1 1 1 1 1 1 1 1;
    1 1 1 2 2 2 2 1 1 1;
    2 1 1 2 2 2 2 1 1 1;
    1 1 1 1 1 1 1 1 2 1
]

# ╔═╡ 0a1842c9-c144-4dd3-8555-e56581bd3f5a
begin
	start_row, start_col, end_row, end_col = trova_indici_pattern(m,2)
	println("starting pattern cell: ($start_row, $start_col)")
	println("ending pattern cell: ($end_row, $end_col)")
end

# ╔═╡ 64b12b8d-f824-497d-a9db-0ab1b7402186
begin
	coords2 = trova_indici_pattern(labels_map(FELZ),2)
	coords3 = trova_indici_pattern(labels_map(FELZ),3)
	coords4 = trova_indici_pattern(labels_map(FELZ),4)
	coords5 = trova_indici_pattern(labels_map(FELZ),5)
end

# ╔═╡ 1ef07af2-32d4-407c-addf-7c38295d46c5
mosaic(
	tray[coords2[1]:coords2[3],coords2[2]:coords2[4]],
	tray[coords3[1]:coords3[3],coords3[2]:coords3[4]],
	tray[coords4[1]:coords4[3],coords4[2]:coords4[4]],
	tray[coords5[1]:coords5[3],coords5[2]:coords5[4]],
	ncol=2, npad=2, fillvalue=1
)

# ╔═╡ d6c1176e-5751-4953-8c5e-b53d0ebf1fef
md"""
## Angle balance
"""

# ╔═╡ 9db76118-ba10-4be9-ad69-7fa4d5d4e89e
# segbox = tray[coords2[1]:coords2[3],coords2[2]:coords2[4]] # example
# segbox = tray[coords3[1]:coords3[3],coords3[2]:coords3[4]] # example
segbox = tray[coords4[1]:coords4[3],coords4[2]:coords4[4]] # example
# segbox = tray[coords5[1]:coords5[3],coords5[2]:coords5[4]] # example

# ╔═╡ aac95cb6-f26b-479a-a8ca-73f5f51fd320
morphogradient(segbox)

# ╔═╡ f1cd8618-15b0-478a-be2f-f777ed63fef7
Float64.(segbox)

# ╔═╡ 15ff0d06-06cd-4c67-bd09-dd9538985e51
ret = adjust_histogram(segbox, ContrastStretching(t = 0.2, slope = 10))

# ╔═╡ 65d7cbdf-d5fe-44b7-9456-cff870a0ecd9
# using ImageBinarization # if we want
# binarize(segbox, algorithm())

# ╔═╡ e94719d7-8148-43dc-a3cb-94359ca58ebd
# segbox_tresh = Gray.(segbox .> 0.2)
segbox_tresh = Gray.(ret .> .5)

# ╔═╡ 9c08b894-5cfc-46c2-b58a-4f8b90c14bae
function find_angle_rot(img)
	corner_row, corner_col = 0, 0
	for i in 1:size(img)[1]
		if 1 in img[i,:]
			corner_row = i
			corner_col = findfirst(x->x==1, img[i,:])
			return (corner_row, corner_col)
		end
	end
end

# ╔═╡ c1c97128-bfe2-4d12-9a64-15b9717a9ed1
find_angle_rot(segbox_tresh)

# ╔═╡ 5319f1d0-08ff-41f3-8ec0-7d93976f9247
teta = atan(-0.12)

# ╔═╡ bbe4fbd1-9bc7-436f-bef4-fd80e00a77c0
imrotate(segbox,teta)

# ╔═╡ d5b651b1-9dd6-44cb-ad37-dcad5ae83c9d
can = Gray{N0f8}.(canny(segbox_tresh, (Percentile(91), Percentile(9))))
# maybe it can be helpful somehow

# ╔═╡ 2f63d6ec-feba-4ffa-8aaf-cb1e468de4e0
Gray.(can)

# ╔═╡ 17f76ea2-9510-49be-9e71-841dce362ebf
typeof(segbox)

# ╔═╡ 638687ec-1b66-4d2c-952e-d580e39ea454
summary(segbox)

# ╔═╡ Cell order:
# ╟─0ca3370e-791c-46bb-8e07-383d254285d5
# ╟─54a14054-cd36-4d95-9a1a-993b874510ed
# ╠═f2166155-6b17-46a6-bf2f-58476a16cb96
# ╠═78269f2f-2e7d-44a5-a998-b57094722e88
# ╠═7855cb01-a95d-47de-8ef6-43b62af38db8
# ╠═0892ac49-4413-4b65-80c5-f40c0390dfd9
# ╠═a57c0c31-1891-4955-bd05-8d89ed578779
# ╠═21704e07-f4e4-436f-b10a-c50ed96dd886
# ╠═330fbf6e-fa52-4c0e-96bc-9a8994f57a09
# ╠═fbb5afcc-4fe9-47e5-8e36-d4e0895a1455
# ╠═18f71bf9-a511-4afe-a11d-027348acbd3d
# ╠═31609e07-ef7e-44e3-b14a-06657ec0886e
# ╠═b4b1cb67-c40b-4905-8a4a-a471a9d36f66
# ╟─5219e0b4-d048-43dd-b752-a4b14824a985
# ╠═52fc6b9a-6e13-4eb7-a256-353187eee650
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
# ╟─346081ac-1caa-40cc-84df-8885c4ce2aba
# ╠═ead3c71b-1027-4048-8833-7ab7eee4e90a
# ╠═213607ff-95f2-46e3-a806-a22d9d5eed08
# ╠═30a8dab9-ab49-4902-8a08-27c1311ba5b8
# ╠═64e7649a-a7ff-4603-9a35-67deb350e363
# ╠═a1850f3a-b491-4f71-8642-bac933417339
# ╠═689ad7ae-e9c4-4481-b634-ac8211df55eb
# ╠═f0e6afd1-3032-4df0-959f-f2a89348c7ef
# ╟─b9adec8d-a9e6-4d24-a64f-a214ef8bab1c
# ╠═7951c08e-e05f-44be-9a07-853f4c3836b0
# ╠═a55dc561-1405-4e92-9bc8-50be622762a7
# ╟─5665c78f-0f26-47a8-9cc7-d936fb410195
# ╠═39d7ffac-00ee-4f85-8850-45e3d883c6cf
# ╠═5cd48d20-a223-4599-9cbb-a9c2dad33336
# ╠═f84f29ea-6642-4243-ad88-f4b6ac6e9809
# ╟─57fbdc8a-86a2-4041-a6bb-659b539c3d2b
# ╠═0d4c8b48-6c77-49c1-b954-4776083b67b5
# ╠═ddda3d0f-5efa-4794-886f-76f3ba7dfba6
# ╠═ec4acc55-68f7-4b80-8101-df4fc6444f3c
# ╟─0e948dac-4c47-48ff-80e8-bcf5b0179992
# ╠═9d934cfb-c26a-4493-b717-0aaf8295237d
# ╠═7f7891a6-14cc-42b8-b724-a39883a2308b
# ╠═0a1842c9-c144-4dd3-8555-e56581bd3f5a
# ╠═64b12b8d-f824-497d-a9db-0ab1b7402186
# ╠═1ef07af2-32d4-407c-addf-7c38295d46c5
# ╟─d6c1176e-5751-4953-8c5e-b53d0ebf1fef
# ╠═9db76118-ba10-4be9-ad69-7fa4d5d4e89e
# ╠═aac95cb6-f26b-479a-a8ca-73f5f51fd320
# ╠═f1cd8618-15b0-478a-be2f-f777ed63fef7
# ╠═15ff0d06-06cd-4c67-bd09-dd9538985e51
# ╠═65d7cbdf-d5fe-44b7-9456-cff870a0ecd9
# ╠═e94719d7-8148-43dc-a3cb-94359ca58ebd
# ╠═9c08b894-5cfc-46c2-b58a-4f8b90c14bae
# ╠═c1c97128-bfe2-4d12-9a64-15b9717a9ed1
# ╠═5319f1d0-08ff-41f3-8ec0-7d93976f9247
# ╠═bbe4fbd1-9bc7-436f-bef4-fd80e00a77c0
# ╠═d5b651b1-9dd6-44cb-ad37-dcad5ae83c9d
# ╠═2f63d6ec-feba-4ffa-8aaf-cb1e468de4e0
# ╠═17f76ea2-9510-49be-9e71-841dce362ebf
# ╠═638687ec-1b66-4d2c-952e-d580e39ea454

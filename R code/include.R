####################################
## Include Me in your scripts with
## source("include.R")
####################################

Mode <- function(x,verbose=0) {
	ux <- unique(x)
	if(verbose){
		print(table(x))
	}
	ux[which.max(tabulate(match(x, ux)))]
}

########################
# useful aliases
########################
size = dim
end = len = lenght = length
assert = stopifnot
ln = log

extrema = function(vec){
	return(c(min(vec),max(vec)))
}

count_na = function(vec){
	na_vals = sum(as.numeric(is.na(vec)))
	return(na_vals)
}
na_count = quanti_na = count_na

########################
# loading procedure, with feedback
########################
library(crayon)
library(hash)

########################
# function to get colors for plotting
########################
library(RColorBrewer)
fun_colori = function(len=2, seed=33, show=1, seed_div = "Set3"){
	hcols_ = hcl.pals()
	if(seed=="rand"){
		seed = round(runif(1,0,115))
		col.ramp_ = hcl.colors(len,palette=hcols_[seed%%115+1])
	}
	if(seed=="div"){ # create a divergent palette
		col.ramp_ = brewer.pal(len, seed_div)
		# possible seed_div choices (and max len supported)
		# Set3	    12
		# Paired    12
		# Pastel1   9
		# Set1	    9
		# Accent    8
		# Dark2     8
		# Pastel2   8
		# Set2	    8
	}
	else{
		col.ramp_ = hcl.colors(len,palette=hcols_[seed%%115+1])
	}
	if(show==1){
		dati_ <- matrix(1:100, ncol = 1)
		par(mar=rep(2,4))
		image(dati_, col = col.ramp_, axes = FALSE)
		title(main=paste("palette",seed,"of",len,"colors"))
		# title(main=paste("palette",seed))
	}
	return(col.ramp_)
	
}
colori_fun = colorami = colora = fun_colori # aliases
# usage: cols = colora(7) for getting a palette with 7 colors
# usage: cols = colora(7,456) for getting a palette of seed 456 with 7 colors

cat(crayon::cyan("Created function to get color palettes. Available as"),crayon::red("colora(len, seed, show).\n"))
cat(crayon::italic("Try for example colora(10,56,1).\n"))



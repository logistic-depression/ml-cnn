def addborders(img,n):
	export = np.zeros(img.shape+np.array(2*n), img.dtype)+0
	# Here, 0 is the value of the added border
	export[n:-n,n:-n] = img
	return export
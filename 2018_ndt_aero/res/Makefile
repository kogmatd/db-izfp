all: mcx.png ocx.png regx.png reg_linex.png reg_line0x.png

mc.png: mcx1.png mcx2.png mcx3.png
	python3 -c 'from skimage import io; import numpy as np; io.imsave("mc.png",np.concatenate([io.imread("mcx%i.png"%i) for i in range(1,4)],axis=1))'

oc.png: ocal.png ocx1.png ocx2.png ocx3.png
	python3 -c 'from skimage import io; import numpy as np; io.imsave("oc.png",np.concatenate([io.imread("ocx%i.png"%i if i>0 else "ocal.png") for i in range(4)],axis=1))'

%.pdf: %.dem %.dat
	gnuplot $*.dem

%.png: %.pdf
	convert -density 600 $< $@

%x.png: %.png
	python3 -c 'from skimage import io; import numpy as np; im=io.imread("$<"); io.imsave("$@",(im[:,:,:3]*(im[:,:,3:4]/255)+255-im[:,:,3:4]).astype(np.uint8));'

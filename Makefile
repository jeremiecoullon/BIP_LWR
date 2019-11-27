solvers:
	f2py -c BIP_LWR/src/lwr_del_Cast.f90 -m lwr_del_Cast;

	mkdir -p BIP_LWR/bin
	mv *.so BIP_LWR/bin/

	if [ -d lwr_del_Cast.cpython-36m-darwin.so.dSYM ]; then rm -Rf lwr_del_Cast.cpython-36m-darwin.so.dSYM; fi
	# rm -r *.so.dSYM


test:
	py.test

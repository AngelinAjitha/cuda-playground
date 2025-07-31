build_poly_eval:
	mkdir -p work
	nvcc -o work/test_poly_eval  test.cu -std=c++17

test_poly_eval: build_poly_eval
	work/test_poly_eval
common_src += files(['CSysSolve_b.cpp',
                     'CSysSolve.cpp',
                     'CSysVector.cpp',
                     'CSysMatrix.cpp',
                     'CPastixWrapper.cpp',
                     'blas_structure.cpp'])

  if get_option('enable-cuda')
    common_src += files(['CSysMatrixGPU.cu', 'CSysVectorGPU.cu',])
  elif get_option('enable-kokkos')
    common_src += files(['CSysMatrixKokkos.cpp', 'CSysVectorKokkos.cpp',])
  endif

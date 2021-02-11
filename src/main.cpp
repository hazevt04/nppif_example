// C++ File for main

/*
 * From https://forums.developer.nvidia.com/t/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/63161
 * This is a working(?) example of how to use nppif from Robert Crovella, the NVIDIA Dev Forum Mod
 * I reformatted it to not make my eyes bleed.
 */

#include <nppi_filtering_functions.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include "kernel.cuh"

int main(){

/**
 * 8-bit unsigned single-channel 1D row convolution.
 */
  bool debug = true;

  const int simgrows = 32;
  const int simgcols = 32;
  Npp8u *d_pSrc, *d_pDst;
  const int nMaskSize = 3;

  NppiSize oROI;  
  oROI.width = simgcols - nMaskSize;  
  oROI.height = simgrows;
  
  const int simgsize = simgrows*simgcols*sizeof(d_pSrc[0]);
  const int dimgsize = oROI.width*oROI.height*sizeof(d_pSrc[0]);

  const int simgpix  = simgrows*simgcols;
  const int dimgpix  = oROI.width*oROI.height;
  
  const int nSrcStep = simgcols*sizeof(d_pSrc[0]);
  const int nDstStep = oROI.width*sizeof(d_pDst[0]);
  
  const int pixval = 1;
  const int nDivisor = 1;
  
  //const Npp32s h_pKernel[nMaskSize] = {pixval, pixval, pixval};
  //Npp32s *d_pKernel;
  
  const Npp32s nAnchor = 2;
  
  cudaError_t err = cudaMalloc((void **)&d_pSrc, simgsize);
  assert(err == cudaSuccess);
  
  err = cudaMalloc((void **)&d_pDst, dimgsize);
  assert(err == cudaSuccess);
  
  //err = cudaMalloc((void **)&d_pKernel, nMaskSize*sizeof(d_pKernel[0]));
  //assert(err == cudaSuccess);
  
  // set image to pixval initially
  err = cudaMemset(d_pSrc, pixval, simgsize);
  assert(err == cudaSuccess);
  
  err = cudaMemset(d_pDst, 0, dimgsize);
  assert(err == cudaSuccess);
  
  //err = cudaMemcpy(d_pKernel, h_pKernel, nMaskSize*sizeof(d_pKernel[0]), cudaMemcpyHostToDevice);
  //assert(err == cudaSuccess);
  
  // copy src to dst
  NppStatus ret =  nppiFilterRow_8u_C1R(d_pSrc, nSrcStep, d_pDst, nDstStep, oROI, d_pKernel, nMaskSize, nAnchor, nDivisor);
  assert(ret == NPP_NO_ERROR);
  
  Npp8u *h_imgres = new Npp8u[dimgpix];
  
  err = cudaMemcpy(h_imgres, d_pDst, dimgsize, cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);
  
  std::cout << "Image Results:\n"; 

  // test for filtering
  for (int i = 0; i < dimgpix; i++) {
    // TODO: Make this not display the ASCII characters
    std::cout << "\th_imgres[" << std::dec << i << "] = " << std::hex << std::showbase << (int)h_imgres[i] << "\n"; 
    assert(h_imgres[i] == (pixval*pixval*nMaskSize));
  }
  std::cout << "\nAll matched expected";
  
  if( h_imgres ) delete [] h_imgres;
  if ( d_pDst ) cudaFree( d_pDst );
  if ( d_pSrc ) cudaFree( d_pSrc );

  return 0;
}

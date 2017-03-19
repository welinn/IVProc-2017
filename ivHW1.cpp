#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;

void histogram(Mat);
void skeleton(Mat);
void findMask(Mat, int*, int*, int, int, int);

int main(int argc, char* argv[]) {

  int choice;
  Mat image;

  if ( argc != 2 ) {
    printf("exe image\n");
    return -1;
  }

  image = imread(argv[1], 1); // > 0 : BGR

  // 檢查影像是否正確讀入
  if ( !image.data ) {
    printf("No image data \n");
    return -1;
  }

  printf("1: histogram\n2: skeleton\n");
  scanf("%d", &choice);

  if(choice == 1) histogram(image);
  else skeleton(image);

  return 0;
}

void histogram(Mat img){
  int segment = 200;
  int mask[] = {1, -2, 1};
  int totalLen = img.cols * img.rows;
  int i, width = img.cols, height = img.rows, mod;
  int hist[segment + 1], no, max = 0;
  double gap = 2 * M_PI / segment;
  double theta[totalLen], pxf, pyf;

  for(i = 0; i < segment; i++) hist[i] = 0;
  for(i = 0; i < totalLen; i++){
    mod = i % width;
    if(mod == 0){
      pxf = mask[0] * img.data[i * 3] +
            mask[1] * img.data[i * 3] +
            mask[2] * img.data[i * 3 + 3];
    }
    else if(mod == width - 1){
      pxf = mask[0] * img.data[i * 3 - 3] +
            mask[1] * img.data[i * 3] +
            mask[2] * img.data[i * 3];
    }
    else{
      pxf = mask[0] * img.data[i * 3 - 3] +
            mask[1] * img.data[i * 3] +
            mask[2] * img.data[i * 3 + 3];
    }

    if(i < width){
      pyf = mask[0] * img.data[i * 3] +
            mask[1] * img.data[i * 3] +
            mask[2] * img.data[(i + width) * 3];
    }
    else if(i / width == height - 1){
      pyf = mask[0] * img.data[(i - width) * 3] +
            mask[1] * img.data[i * 3] +
            mask[2] * img.data[i * 3];
    }
    else{
      pyf = mask[0] * img.data[(i - width) * 3] +
            mask[1] * img.data[i * 3] +
            mask[2] * img.data[(i + width) * 3];
    }
    if(pxf == 0) pxf -= 0.01;
    theta[i] = atan2(pyf, pxf);
    if(theta[i] < 0) theta[i] += M_PI * 2;
  }
  for(i = 0; i < totalLen; i++){
    no = (int) floor(theta[i] / gap);
    hist[no]++;
    if(hist[no] > max) max = hist[no];
  }
  max /= 100;
  Mat plot(max, segment, CV_8U);
  totalLen = segment * max;
  for(i = 0; i < totalLen; i++){
    if(i / segment <= max - hist[i % segment] / 100) plot.data[i] = 0;
    else plot.data[i] = 255;
  }
  namedWindow("Display Image", WINDOW_AUTOSIZE);
  imshow("Display Image", plot);
  waitKey(0);

  return;
}

void skeleton(Mat img){
  int i, j, k, index, doRoop, step = 1;
  int wigth = img.cols, height = img.rows;
  int totalLen = wigth * height;
  int nz5, tz5, mask[10], maskTz[8];
  Mat dst, tmp;

  threshold(img, dst, 128, 255, THRESH_BINARY);
  while(1){
    step *= -1;
    doRoop = 0;
    tmp = dst.clone();
    for(i = 0; i < totalLen; i++){
      nz5 = tz5 = 0;
      if(tmp.data[i * 3] == 0){
        findMask(tmp, mask, maskTz, i, wigth, height);
        for(j = 0; j < 9; j++){
          if(mask[j] == 1) nz5++;
        }
        nz5--;
        if(nz5 < 7 && nz5 > 1){
          for(j = 0; j < 7; j++){
            if(maskTz[j] == 0 && maskTz[j+1] == 1) tz5++;
          }
          if(tz5 == 1){
            if((step > 0 && ((mask[1] == 0 || mask[5] == 0 || mask[7] == 0) && (mask[1] == 0 || mask[3] == 0 || mask[5] == 0))) ||
               (step < 0 && ((mask[3] == 0 || mask[5] == 0 || mask[7] == 0) && (mask[1] == 0 || mask[3] == 0 || mask[7] == 0)))){
              dst.data[i * 3] = dst.data[i * 3 + 1] = dst.data[i * 3 + 2] = 255;
              doRoop++;
            }
          }
        }
      }
    }
    if(doRoop == 0) break;
  }
  imshow("thinning", dst);
  waitKey(0);
}
void findMask(Mat img, int* mask, int* maskTz, int n, int w, int h){
  int i, j, tmp, index;
  for(i = 0; i < 9; i++){
    j = n % w;

    if(n < w && i < 3){ //上邊 + 角
      if((n == 0 && i == 0) || (n == w-1 && i == 2)) index = n * 3;
      else index = (n + (i - 1)) * 3;
    }
    else if(n / w == h-1 && i > 5){ //下邊 + 角
      if((j == 0 && i == 6) || (j == w-1 && i == 8)) index = n * 3;
      else index = (n + (i - 7)) * 3;
    }
    else if((j == 0 && i % 3 == 0) || (j == w-1 && i % 3 == 2)){ //左右邊
      index = (n + w * ((i / 3) - 1)) * 3;
    }
    else{ //非邊點
      index = (n + (w * (i / 3 - 1) + (i % 3 - 1))) * 3;
    }

    mask[i] = img.data[index] == 255 ? 0 : 1;
    if(i < 3) maskTz[i] = mask[i];
    else if(i == 3) maskTz[7] = mask[i];
    else if(i == 5) maskTz[3] = mask[i];
    else if(i == 6) maskTz[6] = mask[i];
    else if(i == 7) maskTz[5] = mask[i];
    else if(i == 8) maskTz[4] = mask[i];
  }
}

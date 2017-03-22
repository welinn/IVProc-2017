#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void onMouse(int, int, int, int, void*);
Mat sampleArr(Mat, int, int);

int def = 0;
int count, attr;//, trai;
float *trainingData, *attribute;
Mat src;

int main(){

  int cc, i, j, x, y;
  int row, col, i3, j3;
  int defaultCount = 20;
  float defaultAttr[] = { 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  printf("How many samples? ( 0 for default )\n");
  scanf("%d", &count);
  if(count == 0){
    count = defaultCount;
    def = 1;
  }
  cc = count;

  trainingData = (float*) malloc(sizeof(float) * cc * 3 * 16);
  if(def) attribute = defaultAttr;
  else attribute = (float*) malloc(sizeof(float) * cc);

  attr = 0;

  src = imread("./hw2.jpg");
  if(src.data == 0){
    printf("no image\n");
    return -1;
  }
  namedWindow("Image");
  imshow("Image",src);
  setMouseCallback("Image", onMouse, NULL);
  waitKey();

  Mat testMat = imread("./hw2-test.jpg");
  row = testMat.rows;
  col = testMat.cols;
  Mat image = Mat::zeros(row, col, CV_8U);
  Mat attrMat(cc, 1, CV_32FC1, attribute);
  Mat trainingDataMat(cc, 48, CV_32FC1, trainingData);

  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  CvSVM svm;
  svm.train(trainingDataMat, attrMat, Mat(), Mat(), params);

  for (i = 0; i < row; i++){
    for (j = 0; j < col; j++){

      Mat sampleMat = sampleArr(testMat, i, j);
      float response = svm.predict(sampleMat);
      if(response == 1){
        image.data[i * col + j] = 255;
      }
      else if(response == -1){
        image.data[i * col + j] = 0;
      }	
    }
  }
  namedWindow("Image");
  imshow("Image",image);
  waitKey();

  free(trainingData);
  free(attribute);
  return 0;
}

void onMouse(int event, int y, int x, int flags, void *param){
  Mat img = src.clone();
  int col = img.cols;
  int i, j, k;
  Rect rect;
  switch (event){
    //滑鼠左鍵
    case CV_EVENT_LBUTTONDOWN:
      //框框左上角座標 & 寬高
      rect = Rect(y, x, 4, 4);

      k = 0;
      x *= (col * 3);
      x += (y * 3);
      for(i = 0; i < 4; i ++){
        for(j = 0; j < 12; j++){
          trainingData[k++] = img.data[x + i * col + j];
        }
      }
      //畫框框
      rectangle(img, rect, Scalar(0, 0, 255), 2);
      namedWindow("Image");
      imshow("Image", img);

      if(!def){
        printf("Enter attribute\n");
        scanf("%f", attribute + attr);
        attr++;
      }
      count--;
      if(count == 0) cvDestroyWindow("Image");

      break;
  }
}

Mat sampleArr(Mat testMat, int x, int y){
  float data[48];
  int i, j, k;
  int col = testMat.cols;

  k = 0;
  x *= (col * 3);
  x += (y * 3);
  for(i = 0; i < 4; i ++){
    for(j = 0; j < 12; j++){
      data[k++] = testMat.data[x + i * col + j];
    }
  }
  Mat m(1, 48, CV_32FC1, data);
  return m;
}

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void onMouse(int, int, int, int, void*);

int def = 0;
int count, attr, trai;
float *trainingData, *attribute;
Mat src;
Rect rect;

int main(){

  int cc, i, j, row, col, i3, j3;
  int defaultCount = 20;
  float defaultAttr[] = { 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  printf("How many samples?\n");
  scanf("%d", &count);
  if(count == 0){
    count = defaultCount;
    def = 1;
  }
  cc = count;

  trainingData = (float*) malloc(sizeof(float) * cc * 3);
  attribute = (float*) malloc(sizeof(float) * cc);
  if(def) attribute = defaultAttr;

  trai = attr = 0;

  src = imread("./hw2.jpg");
  if(src.data == 0){
    printf("no image\n");
    return -1;
  }
  namedWindow("Image");
  imshow("Image",src);
  setMouseCallback("Image", onMouse, NULL);
  waitKey();

  row = src.rows;
  col = src.cols;
  Mat image = Mat::zeros(row, col, CV_8U);
  Mat attrMat(cc, 1, CV_32FC1, attribute);
  Mat trainingDataMat(cc, 3, CV_32FC1, trainingData);
  Mat testMat = imread("./hw2-test.jpg");
//  Mat testMat = src;

  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  CvSVM svm;
  svm.train(trainingDataMat, attrMat, Mat(), Mat(), params);

  //image type = CV_8UC3 : 1 pixel 要3個值
  //Vec3b bl(0, 0, 0), w (255, 255, 255);

  for (i = 0; i < row; i++){
    for (j = 0; j < col; j++){
      i3 = i * 3;
      j3 = j * 3;
      //把BGR的值當array[1][3]塞到Mat裡面
      Mat sampleMat = (Mat_<float>(1,3) << testMat.data[i3 * col + j3],
                                           testMat.data[i3 * col + j3 + 1],
                                           testMat.data[i3 * col + j3 + 2]);
      float response = svm.predict(sampleMat);

      if(response == 1){
        //image.at<Vec3b>(i,j) = w; // 8UC3
        image.data[i * col + j] = 255;
      }
      else if(response == -1){
        //image.at<Vec3b>(i,j) = bl; // 8UC3
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

void onMouse(int event, int x, int y, int flags, void *param){
  Mat img = src.clone();
  switch (event){
    //滑鼠左鍵
    case CV_EVENT_LBUTTONDOWN:
      //框框左上角座標 & 寬高
      rect = Rect(x, y, 4, 4);
      //防止超出原圖片
      //rect &= Rect(0, 0, src.cols, src.rows);

      *(trainingData + trai++) = img.data[x * img.rows * 3 + y];
      *(trainingData + trai++) = img.data[x * img.rows * 3 + y + 1];
      *(trainingData + trai++) = img.data[x * img.rows * 3 + y + 2];

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

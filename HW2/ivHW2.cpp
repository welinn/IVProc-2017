#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat src;
Rect rect;

void onMouse(int event, int x, int y, int flags, void *param)
{  
  Mat img = src.clone();
  switch (event){ 
    //滑鼠左鍵
    case CV_EVENT_LBUTTONDOWN:
      //框框左上角座標 & 寬高  
      rect = Rect(x, y, 4, 4);
      //防止超出原圖片
      rect &= Rect(0, 0, src.cols, src.rows);

      //畫框框 
      rectangle(img, rect, Scalar(0, 0, 255), 2);
      namedWindow("Image");
      imshow("Image", img);

      waitKey(0);

      break;
 
  }  
}  

int main()
{
  src=imread("./hw2.jpg");
  if(src.data == 0){
      printf("no image\n");
      return -1;
  }
  namedWindow("Image");
  imshow("Image",src);
  setMouseCallback("Image", onMouse, NULL);
  waitKey();
  return 0;
}

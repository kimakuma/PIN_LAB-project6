import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
//import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

//import org.opencv.highgui.VideoCapture;

// resize
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.imgcodecs.Imgcodecs.imread;

// rectangle
import org.opencv.core.Point;
import org.opencv.core.Scalar;

// roi
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;

// display
import static org.opencv.highgui.HighGui.waitKey;

public class openn
{
    public static void main(String[] args)
    {
        int capture_width = 1000;
        int capture_height = 550;
        int margin = 50;
        int roi_long = 400;
        int left = margin;
        int right = left + roi_long;
        int top = margin;
        int bottom = top + roi_long;

        System.load("/home/java/opencv/opencv/build/lib/libopencv_java453.so");

        Mat frame = new Mat();
        Mat resized_image = new Mat();
        // Mat roi = new Mat();
        Mat gray = new Mat();
        Mat img_threshold = new Mat();
        Mat digit = new Mat();
        Mat reshape = new Mat();
        Mat reshaped = new Mat();
        Mat unsqueezed = new Mat();

        // cap
        VideoCapture cap = new VideoCapture(0);

        if (!cap.isOpened())
        {
            System.out.println("Can't open the camera");
            System.exit(-1);
        }

        while(true)
        {
            cap.read(frame);

            // resize frame
            Size size = new Size(capture_width, capture_height);
            resize(frame, resized_image, size);

            // rectangle
            Point point1 = new Point(50, 50);
            Point point2 = new Point(450, 450);
            Scalar color = new Scalar(0, 255, 0);
            rectangle (resized_image, point1, point2, color, 2);

            // roi
            Rect roi_size = new Rect(left + 2, top + 2, 396, 396);
            Mat roi = new Mat(resized_image, roi_size);

            // roi_gray
            cvtColor(roi, gray, COLOR_BGR2GRAY); // or RGB2GRAY

            // threshold
            threshold(gray, img_threshold, 150, 255, THRESH_BINARY_INV);

            // text
            putText(resized_image, "Result", new Point(right + margin*3, margin*3), 0, 3, new Scalar(0, 255, 0), 3, 8);
            putText(resized_image, "000", new Point(right + margin * 5 - 20, margin * 7 + 20), 0, 6, new Scalar(0, 255, 0), 5, 8);

            // display
            HighGui.imshow("camera", resized_image);
            HighGui.imshow("roi", img_threshold);
            //resized_image.setVisible(true);

            // HighGui.waitKey(15000);
            //System.exit(0);
            if (waitKey(1) == 27)
            {
                System.out.println("Exit \n");
                break;
            }
        }
    }
}

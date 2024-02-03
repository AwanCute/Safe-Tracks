package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class objectDetectorClass {
    // should start from small letter

    // this is used to load model and predict
    private Interpreter interpreter;
    // store all label in array
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3; // for RGB
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // use to initialize gpu in app
    private GpuDelegate gpuDelegate;
    private int height=0;
    private  int width=0;
    public CameraActivity cameraActivity;

    public objectDetectorClass(CameraActivity cameraActivity, AssetManager assetManager, String modelPath, String labelPath, int inputSize) throws IOException{
        this.cameraActivity = cameraActivity;
        INPUT_SIZE=inputSize;

        // use to define gpu or cpu // no. of threads
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // set it according to your phone
        // loading model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        // load labelmap
        labelList=loadLabelList(assetManager,labelPath);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        // to store label
        List<String> labelList=new ArrayList<>();
        // create a new reader
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        // loop through each line and store it to labelList
        while ((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    // create new Mat function
    public Mat recognizeImage(Mat mat_image){
        // Rotate original image by 90 degree get get portrait frame
        Mat rotated_mat_image=new Mat();
        Core.flip(mat_image.t(),rotated_mat_image,1);
        // if you do not do this process you will get improper prediction, less no. of object
        // now convert it to bitmap
        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);
        // define height and width
        height=bitmap.getHeight();
        width=bitmap.getWidth();

        // scale the bitmap to input size of model
        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

         // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);

        // defining output
        // 10: top 10 object detected
        // 4: there coordinate in image
      //  float[][][]result=new float[1][10][4];
        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();
        // we are not going to use this method of output
        // instead we create treemap of three array (boxes,score,classes)

        float[][][]boxes =new float[1][10][4];
        // 10: top 10 object detected
        // 4: there coordinate in image
        float[][] scores=new float[1][10];
        // stores scores of 10 object
        float[][] classes=new float[1][10];
        // stores class of object

        // add it to object_map;
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        // now predict
        interpreter.runForMultipleInputsOutputs(input,output_map);

        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        // loop through each object
        // as output has only 10 boxes

        List<String> classesToDetect = Arrays.asList("chair", "bench", "train", "person");
        int personCount = 0, obstacleCounter = 0;
        boolean personDetected = false, personOnLeft = false, personOnRight = false, personOnFront = false;
        boolean trainDetected = false, closeToTrain = false;
        boolean obstacleDetected = false, obstacleOnLeft = false, obstacleOnRight = false, obstacleOnFront = false;

        float minSizeThreshold = 0.25f * width * height; // 25% of the screen area
        float closeToTheTrain = 0.8f * width * height;

        for (int i=0;i<10;i++){
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);
            // define threshold for score
            if(score_value>0.5 && !cameraActivity.isDeviceInLandscape && cameraActivity.textToSpeechInitialized){
                Object box1=Array.get(Array.get(value,0),i);
                // we are multiplying it with Original height and width of frame

                float top=(float) Array.get(box1,0)*height;
                float left=(float) Array.get(box1,1)*width;
                float bottom=(float) Array.get(box1,2)*height;
                float right=(float) Array.get(box1,3)*width;

                String label = labelList.get((int) class_value);

                // Calculate bounding box size
                float boxWidth = right - left;
                float boxHeight = bottom - top;
                float boxSize = boxWidth * boxHeight;

                // Log the size of the bounding box
                Log.d("ObjectDetectionBoundingBoxSize", "Bounding Box Size: " + boxSize);
                Log.d("ObjectDetectionBoundingBoxSize", "Min Size Threshold: " + minSizeThreshold);

                if(boxSize >= closeToTheTrain)
                {
                    if(label.equals("train")) {
                        closeToTrain = true;
                        trainDetected = true;
                    }
                }

                // Check if the bounding box size falls within the range && if the device is in portrait mode
                if (boxSize >= minSizeThreshold) {
                    // This object is considered 'close'

                    //if it detects a train, a person and either a bench or chair
                    if(label.equals("train") && label.equals("person") && (label.equals("chair") || label.equals("bench"))) {
                        trainDetected = true;
                        personDetected = true;
                        obstacleDetected = true;
                    }
                    else if(label.equals("train") && label.equals("person")){
                        trainDetected = true;
                        personDetected = true;
                    }
                    else if(label.equals("train") && (label.equals("chair") || label.equals("bench"))){
                        trainDetected = true;
                        obstacleDetected = true;
                    }
                    else if (label.equals("train")) {
                        trainDetected = true;
                    }
                    else if (label.equals("person") && (label.equals("chair") || label.equals("bench"))){
                        personDetected = true;
                        obstacleDetected = true;
                    }
                    else if (label.equals("person")) {
                        personCount++;
                        personDetected = true;
                        float centerX = (left + right) / 2;

                        if (centerX < width / 3) {
                            personOnLeft = true;
                        } else if (centerX > 2 * width / 3) {
                            personOnRight = true;
                        } else {
                            personOnFront = true;
                        }
                    }
                    else if (label.equals("chair") || label.equals("bench")){
                        obstacleDetected = true;
                        obstacleCounter ++;
                        float centerX = (left + right) / 2;

                        if (centerX < width / 3) {
                            obstacleOnLeft = true;
                        } else if (centerX > 2 * width / 3) {
                            obstacleOnRight = true;
                        } else {
                            obstacleOnFront = true;
                        }
                    }
                }
                if (classesToDetect.contains(label)) {

                    String labelWithScore = label + ": " + String.format("%.2f", score_value * 100) + "%";
                    // draw rectangle in Original frame
                    Imgproc.rectangle(rotated_mat_image,new Point(left,top),new Point(right,bottom),new Scalar(0, 255, 0, 255),2);
                    // write text on frame
                    Imgproc.putText(rotated_mat_image,labelWithScore,new Point(left,top),3,1,new Scalar(255, 0, 0, 255),2);
                }
            }
        }
        if(trainDetected && personDetected && obstacleDetected){
            cameraActivity.speakOut("The train has arrived. Be Careful when boarding, there are people and obstacles near you.");
        }
        else if(trainDetected && personDetected){
            cameraActivity.speakOut("The train has arrived. Be Careful when boarding, there are people near you.");
        }
        else if(trainDetected && obstacleDetected){
            cameraActivity.speakOut("The train has arrived. Be Careful when boarding, there are obstacle near you.");
        }
        else if (trainDetected && closeToTrain){
            cameraActivity.speakOut("You are very close to the train, please be careful of the platform gap");
        }
        else if(trainDetected){
            cameraActivity.speakOut("The train has arrived.");
        }
        else if (personDetected && obstacleDetected){
            cameraActivity.speakOut("Be careful when navigating ahead, there are people and obstacles near you.");
        }
        //If it detects more than 1 person
        else if (personCount > 1 && personDetected) {
            if(personOnLeft && personOnRight || personOnLeft && personOnFront || personOnRight && personOnFront){
                cameraActivity.speakOut("Be careful when navigating ahead, it is crowded.");
            }
            else if (personOnLeft) {
                cameraActivity.speakOut("Be careful, there are people to your left.");
            }
            else if (personOnRight) {
                cameraActivity.speakOut("Be careful, there are people to your right.");
            }
            else if (personOnFront) {
                cameraActivity.speakOut("Be careful, there are people in front of you.");
            }
        }
        //If it detects only 1 person
        else if (personCount == 1 && personDetected) {
            if (personOnLeft) {
                cameraActivity.speakOut("Be careful, there is someone to your left.");
            }
            else if (personOnRight) {
                cameraActivity.speakOut("Be careful, there is someone to your right.");
            }
            else if (personOnFront) {
                cameraActivity.speakOut("Be careful, there is someone in front of you.");
            }
        }
        else if (obstacleCounter > 1 && obstacleDetected){
            if(obstacleOnLeft && obstacleOnRight || obstacleOnLeft && obstacleOnFront || obstacleOnRight && obstacleOnFront){
                cameraActivity.speakOut("Be careful when navigating ahead, it is crowded.");
            }
            else if (obstacleOnLeft) {
                cameraActivity.speakOut("Be careful, there are obstacles to your left.");
            }
            else if (obstacleOnRight) {
                cameraActivity.speakOut("Be careful, there are obstacles to your right.");
            }
            else if (obstacleOnFront) {
                cameraActivity.speakOut("Be careful, there are obstacles in front of you.");
            }
        }
        else if (obstacleCounter == 1 && obstacleDetected) {
            if (obstacleOnLeft) {
                cameraActivity.speakOut("Be careful, there is an obstacle to your left.");
            }
            else if (obstacleOnRight) {
                cameraActivity.speakOut("Be careful, there is an obstacle to your right.");
            }
            else if (obstacleOnFront) {
                cameraActivity.speakOut("Be careful, there is an obstacle in front of you.");
            }
        }
        // select device and run

        // before returning rotate back by -90 degree
        Core.flip(rotated_mat_image.t(),mat_image,0);
        return mat_image;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        // some model input should be quant=0  for some quant=1
        // for this quant=0

        int quant=0;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
    return byteBuffer;
    }
}

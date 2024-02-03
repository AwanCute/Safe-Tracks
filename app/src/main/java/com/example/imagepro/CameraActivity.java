package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.Locale;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Vibrator;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2, SensorEventListener{
    private static final String TAG="MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private objectDetectorClass objectDetectorClass;
    private TextToSpeech textToSpeech;
    private long lastAnnouncementTime = 0;
    private static final long COOLDOWN_PERIOD = 5000; // 5 seconds in milliseconds

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor magnetometer;

    // Declare a Handler
    private Handler handler = new Handler();
    // Define a delay for checking the orientation (e.g., every 100 milliseconds)
    private static final long ORIENTATION_CHECK_DELAY = 100;
    public boolean isDeviceInLandscape;
    private Vibrator vibrator;
    public boolean textToSpeechInitialized;
    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        textToSpeechInitialized = false;
        isDeviceInLandscape = false;
        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // Initialize the sensor manager and sensors
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        try{
            // input size is 300 for this model
            objectDetectorClass=new objectDetectorClass(this, getAssets(),"ssd_mobilenet.tflite","labelmap.txt",300);
            Log.d("MainActivity","Model is successfully loaded");
        }
        catch (IOException e){
            Log.d("MainActivity","Getting some error");
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        initializeTTS();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
        // Register the sensor listeners
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    private void initializeTTS() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int result = textToSpeech.setLanguage(Locale.US);
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e("TTS", "Language not supported");
                } else {
                    textToSpeechInitialized = true;
                    Log.d("TTS", "Text-to-Speech engine is initialized");
                }
            } else {
                Log.e("TTS", "Initialization failed");
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
        // Unregister the sensor listeners to conserve resources
        sensorManager.unregisterListener(this);
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor == accelerometer) {
            // Get accelerometer data
            float[] accelerometerValues = event.values;

            // Check if the device is upright
            if (Math.abs(accelerometerValues[0]) <= 9.81 * Math.sin(Math.toRadians(45))) {
                isDeviceInLandscape = false;

                // Check if the device is pointing upward or downward
                if (Math.abs(accelerometerValues[2]) > 9.81 * Math.sin(Math.toRadians(45))) {
                    isDeviceInLandscape = true;
                    handler.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            if (isDeviceInLandscape) {
                                if (accelerometerValues[2] < 0) {
                                    speakOut("Your phone is pointing upwards. Please ensure your device is facing forward.");
                                } else {
                                    speakOut("Your phone is pointing downwards. Please ensure it's upright.");
                                }
                            }
                        }
                    }, ORIENTATION_CHECK_DELAY);
                }
                else {
                    // Device is neither pointing upwards nor downwards, remove any pending speak requests
                    handler.removeCallbacksAndMessages(null);
                }
            }
            // Device is tilted, prompt to rotate to portrait mode
            else {
                isDeviceInLandscape = true;
                handler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        if (isDeviceInLandscape) {
                            speakOut("Device is tilted. Please rotate your device to portrait mode.");
                        }
                    }
                }, ORIENTATION_CHECK_DELAY);
            }
        } else if (event.sensor == magnetometer) {
            // Get magnetometer data
            float[] magnetometerValues = event.values;
            // ...
        }
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do nothing for now
    }

    private boolean isDeviceInLandscape(float[] accelerometerValues) {
        // Check if the device is in landscape mode based on accelerometer values
        float x = accelerometerValues[0];
        float y = accelerometerValues[1];
        float z = accelerometerValues[2];

        // Define threshold values for tilt detection
        float tiltThreshold = 45.0f;  // Adjust as needed

        // Calculate the tilt angle from accelerometer data
        double tiltAngle = Math.toDegrees(Math.atan2(x, Math.sqrt(y * y + z * z)));

        // Check if the device is tilted significantly to the left or right
        return Math.abs(tiltAngle) > tiltThreshold;
    }


    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba=inputFrame.rgba();
        //mGray=inputFrame.gray();
        // Before watching this video please watch previous video of loading tensorflow lite model
        drawGrid(mRgba);

        // now call that function
        Mat out=new Mat();
        out=objectDetectorClass.recognizeImage(mRgba);

        return out;
    }

    private void drawGrid(Mat mat) {
        int cols = mat.cols();
        int rows = mat.rows();

        // Calculate the coordinates for the horizontal lines
        int rowThird = rows / 3;
        int rowTwoThirds = rowThird * 2;

        // Draw horizontal lines
        Imgproc.line(mat, new Point(0, rowThird), new Point(cols, rowThird), new Scalar(255, 0, 0, 255), 3);
        Imgproc.line(mat, new Point(0, rowTwoThirds), new Point(cols, rowTwoThirds), new Scalar(255, 0, 0, 255), 3);
    }


    public void speakOut(String text) {
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastAnnouncementTime > COOLDOWN_PERIOD && !isDeviceInLandscape) {
            runOnUiThread(() -> {
                textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
                vibrate(200);
                Log.d("TTSDOG", "Speaking: " + text);
            });
            lastAnnouncementTime = currentTime;
        }
        else if(!isSpeaking() && isDeviceInLandscape) {
                runOnUiThread(() -> {
                    textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
                    vibrate(200);
                    Log.d("TTSDOG", "Speaking: " + text);
                });
            }
            //Log.d("TTSDOG", "Cooldown period, skipping announcement");
        }

    // Method to check if Text-to-Speech is currently speaking
    public boolean isSpeaking() {
        return textToSpeech.isSpeaking();
    }

    private void vibrate(long milliseconds) {
        if (vibrator != null) {
            vibrator.vibrate(milliseconds);
        }
    }

}
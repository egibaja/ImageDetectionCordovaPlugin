package com.cloudoki.imagedetectionplugin;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.os.Build;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Base64;
import android.util.Log;
import android.view.Gravity;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.content.Context;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaInterface;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CordovaWebView;
import org.apache.cordova.PluginResult;
import org.json.JSONArray;
import org.json.JSONException;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.core.KeyPoint;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.lang.StringBuilder;


public class ImageDetectionPlugin extends CordovaPlugin implements SurfaceHolder.Callback {

    private static final String  TAG = "OpenCV::Activity";
    private static final int     REQUEST_CAMERA_PERMISSIONS = 133;
    private static final int CAMERA_ID_ANY   = -1;
    private static final int CAMERA_ID_BACK  = 99;
    private static final int CAMERA_ID_FRONT = 98;

    @SuppressWarnings("deprecation")
    private Camera               camera;
    private Activity             activity;
    private SurfaceHolder        surfaceHolder;
    private Mat                  mYuv;
    private Mat                  desc2;
    private Mat                  desc3;
    private FeatureDetector      orbDetector;
    private DescriptorExtractor  orbDescriptor;
    private MatOfKeyPoint        kp2;
    private MatOfKeyPoint        kp3;
    private MatOfDMatch          matches;
    private CallbackContext      cb;
    private Date                 last_time;
    private boolean processFrames = true, thread_over = true, debug = true,
            called_success_detection = false, called_failed_detection = true,
            previewing = false, save_files = false;
    private List<Integer> detection = new ArrayList<>();

    private List<Mat> triggers = new ArrayList<>();
    private List<MatOfKeyPoint> triggers_kps = new ArrayList<>();
    private List<Mat> triggers_descs = new ArrayList<>();
    private int trigger_size = -1, detected_index = -1;

    private double timeout = 0.0;
    private int cameraId = -1;
    private int mCameraIndex = CAMERA_ID_ANY;

    private BaseLoaderCallback mLoaderCallback;
    private FrameLayout cameraFrameLayout;

    private  int count = 0;
    private String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    
    @SuppressWarnings("deprecation")
    private static class JavaCameraSizeAccessor implements CameraBridgeViewBase.ListItemAccessor {

        @Override
        public int getWidth(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.width;
        }

        @Override
        public int getHeight(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.height;
        }
    }

    @Override
    public void initialize(CordovaInterface cordova, CordovaWebView webView) {
        activity = cordova.getActivity();

        super.initialize(cordova, webView);
        Log.e(TAG, "Initializing.....");

        final CordovaInterface cord = cordova;

        mLoaderCallback = new BaseLoaderCallback(activity) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                    {
                        Log.i(TAG, "OpenCV loaded successfully");
                        Context context = cord.getActivity().getApplicationContext(); 
                        final String combA = loadAssetTextAsString(context, "www/combA.txt");
                        final String combB = loadAssetTextAsString(context, "www/combB.txt");
                        cord.getThreadPool().execute(new Runnable() { 
                            public void run() { 

                                // Initialize the patterns to detect
                                final JSONArray patterns = new JSONArray();
                                Log.e(TAG, "asset string cargados, ");
                                patterns.put(combA);
                                patterns.put(combB);
                                Log.e(TAG, "poniendo patterns");
                                triggers.clear();
                                triggers_kps.clear();
                                triggers_descs.clear();
                                Log.e(TAG, "limpiando triggers");

                                String message = "Pattens to be set - " + patterns.length();
                                message += "\nBefore set pattern " + triggers.size();
                                Log.e(TAG,message);
                                setBase64Pattern(patterns);
                                message += "\nAfter set pattern " + triggers.size();
                                Log.e(TAG,message);
                                if(patterns.length() == triggers.size()) {
                                    trigger_size = triggers.size();
                                    message += "\nPatterns set - " + triggers.size();
                                    Log.e(TAG,message);
                                } else {
                                    message += "\nOne or more patterns failed to be set.";
                                    Log.e(TAG,message);
                                }
                            }
                        });

                    } break;
                    default:
                    {
                        super.onManagerConnected(status);
                    } break;
                }
            }
        };

        activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        activity.getWindow().setFormat(PixelFormat.TRANSLUCENT);
        activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        SurfaceView surfaceView = new SurfaceView(activity.getApplicationContext());

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
                Gravity.CENTER);

        cameraFrameLayout = new FrameLayout(activity.getApplicationContext());

        cameraFrameLayout.addView(surfaceView);

        activity.getWindow().addContentView(cameraFrameLayout, params);

        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(this);

        sendViewToBack(cameraFrameLayout);

        setCameraIndex(CAMERA_ID_BACK);
        openCamera();
        Log.e(TAG, "Algo de poner el cameraframelayout a invisible");
        cameraFrameLayout.setVisibility(View.INVISIBLE);        

    }

    @Override
    public boolean execute(String action, JSONArray data,
                           CallbackContext callbackContext) throws JSONException {

        if (action.equals("greet")) {
            Log.i(TAG, "greet called");
            String name = data.getString(0);
            if(name != null && !name.isEmpty()) {
                String message = "Hello, " + name;
                callbackContext.success(message);
            } else {
                callbackContext.error("");
            }
            return true;
        }
        if (action.equals("isDetecting")) {
            Log.i(TAG, "isDetecting called");
            cb = callbackContext;
            return true;
        }
        if(action.equals("startProcessing")) {
            Log.i(TAG, "startProcessing called");
            String message;
            boolean argVal;
            try {
                argVal = data.getBoolean(0);
            } catch (JSONException je) {
                argVal = true;
                Log.e(TAG, je.getMessage());
            }
            if(argVal) {
                processFrames = true;
                message = "Frame processing set to 'true'";
                callbackContext.success(message);
            } else {
                processFrames = false;
                message = "Frame processing set to 'false'";
                callbackContext.error(message);
            }
            return true;
        }
        if(action.equals("setDetectionTimeout")) {
            Log.i(TAG, "setDetectionTimeout called");
            String message;
            double argVal;
            try {
                argVal = data.getDouble(0);
            } catch (JSONException je) {
                argVal = -1;
                Log.e(TAG, je.getMessage());
            }
            if(argVal >= 0) {
                timeout = argVal;
                message = "Processing timeout set to " + timeout;
                callbackContext.success(message);
            } else {
                message = "No value or timeout value negative.";
                callbackContext.error(message);
            }
            return true;
        }
        return false;
    }

    @Override
    public void onStart()
    {
        super.onStart();

        Log.e(TAG, "onStart(): Activity starting");

        if(!checkCameraPermission()) {
            ActivityCompat.requestPermissions(activity,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSIONS);
        }

        if(save_files) {
            int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

            if (permission != PackageManager.PERMISSION_GRANTED) {
                // We don't have permission so prompt the user
                int REQUEST_EXTERNAL_STORAGE = 1;
                ActivityCompat.requestPermissions(
                        activity,
                        PERMISSIONS_STORAGE,
                        REQUEST_EXTERNAL_STORAGE
                );
            }
        }

        thread_over = true;
        debug = true;
        called_success_detection = false;
        called_failed_detection = true;

        last_time = new Date();

        new android.os.Handler().postDelayed(
            new Runnable() {
                public void run() {
                    cameraFrameLayout.setVisibility(View.VISIBLE);
                    cameraFrameLayout.invalidate();
                }
            }, 2000);
    }

    public static void sendViewToBack(final View child) {
        final ViewGroup parent = (ViewGroup)child.getParent();
        if (null != parent) {
            parent.removeView(child);
            parent.addView(child, 0);
        }
    }

    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onPause(boolean multitasking) {
        super.onPause(multitasking);
    }

    @Override
    public void onResume(boolean multitasking) {
        Log.e(TAG, "onResume");
        super.onResume(multitasking);
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, activity, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        if (camera == null) {
            openCamera();
        }
    }

    @Override
    public void onStop() {
        super.onStop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.d(TAG, "Inicializando descriptores y keypoints");
        matches = new MatOfDMatch();
        orbDetector = FeatureDetector.create(FeatureDetector.ORB);
        orbDescriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        kp2 = new MatOfKeyPoint();
        desc2 = new Mat();
        kp3 = new MatOfKeyPoint();
        desc3 = new Mat();
        Log.d(TAG, "Fin de inicializando descriptores y keypoints");
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int height, int width) {
        if(previewing){
            camera.stopPreview();
            previewing = false;
        }

        if (camera != null){
            boolean result = initializeCamera(height, width);
            if( !result ) {
                AlertDialog.Builder builder = new AlertDialog.Builder(activity);
                builder.setTitle("An error occurred")
                        .setMessage("An error occurred while trying to open the camera.")
                        .setCancelable(false)
                        .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                activity.finish();
                            }
                        });
                AlertDialog alert = builder.create();
                alert.show();
            }
            previewing = true;
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
            previewing = false;
        }
    }

    private void setCameraIndex(int index) {
        mCameraIndex = index;
    }


    @SuppressWarnings("deprecation")
    private void openCamera() {
        Log.e(TAG, "Open camera");
        camera = null;

        if (mCameraIndex == CAMERA_ID_ANY) {
            Log.e(TAG, "Trying to open camera with old open()");
            try {
                camera = Camera.open();
            } catch (Exception e) {
                Log.e(TAG, "Camera is not available (in use or does not exist): " + e.getLocalizedMessage());
            }

            if (camera == null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                boolean connected = false;
                for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); camIdx++) {
                    Log.e(TAG, "Trying to open camera with new open(" + camIdx + ")");
                    try {
                        camera = Camera.open(camIdx);
                        connected = true;
                    } catch (RuntimeException e) {
                        Log.e(TAG, "Camera #" + camIdx + "failed to open: " + e.getLocalizedMessage());
                    }
                    if (connected) break;
                }
            }
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                int localCameraIndex = mCameraIndex;
                if (mCameraIndex == CAMERA_ID_BACK) {
                    Log.e(TAG, "Trying to open back camera");
                    Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                    for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); camIdx++) {
                        Camera.getCameraInfo(camIdx, cameraInfo);
                        if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                            localCameraIndex = camIdx;
                            break;
                        }
                    }
                } else if (mCameraIndex == CAMERA_ID_FRONT) {
                    Log.e(TAG, "Trying to open front camera");
                    Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                    for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); camIdx++) {
                        Camera.getCameraInfo(camIdx, cameraInfo);
                        if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                            localCameraIndex = camIdx;
                            break;
                        }
                    }
                }
                if (localCameraIndex == CAMERA_ID_BACK) {
                    Log.e(TAG, "Back camera not found!");
                } else if (localCameraIndex == CAMERA_ID_FRONT) {
                    Log.e(TAG, "Front camera not found!");
                } else {
                    Log.e(TAG, "Trying to open camera with new open(" + localCameraIndex + ")");
                    try {
                        camera = Camera.open(localCameraIndex);
                    } catch (RuntimeException e) {
                        Log.e(TAG, "Camera #" + localCameraIndex + " failed to open: " + e.getLocalizedMessage());
                    }
                }
                cameraId = localCameraIndex;
            }
        }
    }

    @SuppressWarnings("deprecation")
    private boolean initializeCamera(int height, int width) {
        Log.e(TAG, "Initialize java camera");
        boolean result = true;
        synchronized (this) {
            if (camera == null)
                return false;

            /* Now set camera parameters */
            try {
                Camera.Parameters params = camera.getParameters();
                Log.d(TAG, "getSupportedPreviewSizes()");
                List<Camera.Size> sizes = params.getSupportedPreviewSizes();

                if (sizes != null) {
                    /* Select the size that fits surface considering maximum size allowed */
                    Size frameSize = calculateCameraFrameSize(sizes, new JavaCameraSizeAccessor(), height, width);

                    params.setPreviewFormat(ImageFormat.NV21);
                    Log.d(TAG, "Set preview size to " + frameSize.width + "x" + frameSize.height);
                    params.setPreviewSize((int)frameSize.width, (int)frameSize.height);

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.ICE_CREAM_SANDWICH && !android.os.Build.MODEL.equals("GT-I9100"))
                        params.setRecordingHint(true);

                    List<String> FocusModes = params.getSupportedFocusModes();
                    if (FocusModes != null && FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO))
                    {
                        Log.d(TAG, "Set focus mode continuous video " + Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO );
                        params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                    }
                    else if(FocusModes != null && FocusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                        Log.d(TAG, "Set focus mode auto " + Camera.Parameters.FOCUS_MODE_AUTO );
                        params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                    }

                    if(activity != null) {
                        Camera.CameraInfo info = new Camera.CameraInfo();
                        Camera.getCameraInfo(cameraId, info);
                        int cameraRotationOffset = info.orientation;

                        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
                        int degrees = 0;
                        switch (rotation) {
                            case Surface.ROTATION_0:
                                degrees = 0;
                                break; // Natural orientation
                            case Surface.ROTATION_90:
                                degrees = 90;
                                break; // Landscape left
                            case Surface.ROTATION_180:
                                degrees = 180;
                                break;// Upside down
                            case Surface.ROTATION_270:
                                degrees = 270;
                                break;// Landscape right
                        }
                        int displayRotation;
                        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                            displayRotation = (cameraRotationOffset + degrees) % 360;
                            displayRotation = (360 - displayRotation) % 360; // compensate the mirror
                        } else { // back-facing
                            displayRotation = (cameraRotationOffset - degrees + 360) % 360;
                        }

                        Log.v(TAG, "rotation cam / phone = displayRotation: " + cameraRotationOffset + " / " + degrees + " = "
                                + displayRotation);

                        camera.setDisplayOrientation(displayRotation);

                        int rotate;
                        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                            rotate = (360 + cameraRotationOffset + degrees) % 360;
                        } else {
                            rotate = (360 + cameraRotationOffset - degrees) % 360;
                        }

                        Log.v(TAG, "screenshot rotation: " + cameraRotationOffset + " / " + degrees + " = " + rotate);

                        params.setRotation(rotate);

                        params.setPreviewFrameRate(24);// set camera preview

                        camera.setParameters(params);
                        camera.setPreviewDisplay(surfaceHolder);
                        camera.setPreviewCallback(previewCallback);
                    }

                    /* Finally we are ready to start the preview */
                    Log.d(TAG, "startPreview");
                    camera.startPreview();
                }
                else
                    result = false;
            } catch (Exception e) {
                result = false;
                e.printStackTrace();
            }
        }

        return result;
    }

    private Size calculateCameraFrameSize(List<?> supportedSizes, CameraBridgeViewBase.ListItemAccessor accessor, int surfaceHeight, int surfaceWidth) {
        int calcWidth = 0;
        int calcHeight = 0;

        for (Object size : supportedSizes) {
            int width = accessor.getWidth(size);
            int height = accessor.getHeight(size);

            if (width <= surfaceWidth && height <= surfaceHeight) {
                if (width >= calcWidth && height >= calcHeight) {
                    calcWidth = width;
                    calcHeight = height;
                }
            }
        }

        return new Size(calcWidth, calcHeight);
    }

    @SuppressWarnings("deprecation")
    private final Camera.PreviewCallback previewCallback = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {

            Date current_time = new Date();
            double time_passed = Math.abs(current_time.getTime() - last_time.getTime())/1000.0;

            boolean hasTriggerSet = false;
            if(!triggers.isEmpty()){
                hasTriggerSet = triggers.size() == trigger_size;
            }

            if(processFrames && time_passed > timeout && hasTriggerSet) {
                if (thread_over) {
                    thread_over = false;

                    if (mYuv != null) mYuv.release();
                    Camera.Parameters params = camera.getParameters();
                    mYuv = new Mat(params.getPreviewSize().height, params.getPreviewSize().width, CvType.CV_8UC1);
                    mYuv.put(0, 0, data);
                    processFrame(triggers);
                }
                //update time and reset timeout
                last_time = current_time;
                timeout = 0.0;
            }

        }
    };

    private void processFrame(List<Mat> triggers) {

        for (int i = 0; i < triggers.size(); i++){
            final Mat pattern = triggers.get(i);
            final MatOfKeyPoint kp1 = triggers_kps.get(i);
            final Mat desc1 = triggers_descs.get(i);
            final int index = i;

            cordova.getThreadPool().execute(new Runnable() {
                public void run() {
                    Log.e(TAG, "Dentro del thread-------------");
                    int h_init = 0;
                    int h_end = 0; 
                    if (index==0){
                        h_init = 0;
                        h_end = (int)(mYuv.rows() * 0.3);
                    }
                    if (index==1){
                        h_init = (int)(mYuv.rows() * 0.7);
                        h_end = mYuv.rows();
                    }
                    
                    Mat gray = mYuv.submat(h_init, h_end, 0, mYuv.cols()).t();
                    Core.flip(gray, gray, 1);
                    DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE );

                    if(save_files) {
                        Log.e(TAG, "Saving preview image");
                        if (count % 10 == 0) {
                            String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
                            Imgcodecs.imwrite(extStorageDirectory + "/pic" + count + ".png", gray);
                            Log.i("### FILE ###", "File saved to " + extStorageDirectory + "/pic" + count + ".png");
                        }
                        count++;
                    }

                    //Imgproc.equalizeHist(gray, gray);
                    /*
                    String extStorageDirectory = Environment.getExternalStorageDirectory().toString();

                    try{
                        File tempFile = File.createTempFile("config" + index, ".yml", new File(extStorageDirectory));
                        String settings = "%YAML:1.0\nhessianThreshold: 1000\n";

                        FileWriter writer = new FileWriter(tempFile, false);
                        writer.write(settings);
                        writer.close();
                        Log.e(TAG, "cargando...");
                        orbDetector.read(tempFile.getPath());
                    }catch(IOException e){
                        Log.e(TAG, "error cargando la configuracion de SURF");
                        e.printStackTrace();
                    }
                    */

                    if (index==0){
                        orbDetector.detect(gray, kp2);
                        orbDescriptor.compute(gray, kp2, desc2);
                        Log.e(TAG, "Tratando la escena parte superior, que tiene " + kp2.size() + " puntos de interes.");

                        Log.e(TAG, "compute ejecutado...match?");

                        if (!desc1.empty() && !desc2.empty()) {
                            Log.e(TAG, "Se ha producido un match");
                            matcher.match(desc1, desc2, matches);
                            List<DMatch> matchesList = matches.toList();
                            Log.e(TAG, "Hemos encontrado " + matches.size() + " entre la template y la parte alta");
                            LinkedList<DMatch> good_matches = new LinkedList<>();
                            MatOfDMatch gm = new MatOfDMatch();

                            double minDistance = 1000;

                            int rowCount;

                            if(desc1.rows() < matchesList.size())
                                rowCount = desc1.rows();
                            else
                                rowCount = matchesList.size();

                            for (int i = 0; i < rowCount; i++) {
                                double dist = matchesList.get(i).distance;
                                if (dist < minDistance) {
                                    minDistance = dist;
                                }
                            }

                            LinkedList<DMatch> good_matches_reduced = new LinkedList<>();
                            MatOfDMatch gmr = new MatOfDMatch();
                            double upperBound = 2 * minDistance;
                            for (int i = 0; i < rowCount; i++) {
                                if (matchesList.get(i).distance < upperBound && good_matches.size() < 500) {
                                    good_matches.addLast(matchesList.get(i));
                                    if(i < 10 && debug)
                                    {
                                        good_matches_reduced.addLast(matchesList.get(i));
                                    }
                                }
                            }
                            gm.fromList(good_matches);
                            if(debug) {
                                gmr.fromList(good_matches_reduced);
                            }
                        }
                    }else{
                        orbDetector.detect(gray, kp3);
                        orbDescriptor.compute(gray, kp3, desc3);
                        Log.e(TAG, "Tratando la escena parte superior, que tiene " + kp3.size() + " puntos de interes.");

                        Log.e(TAG, "compute ejecutado...match?");

                        if (!desc1.empty() && !desc3.empty()) {
                            Log.e(TAG, "Se ha producido un match");
                            matcher.match(desc1, desc3, matches);
                            List<DMatch> matchesList = matches.toList();
                            Log.e(TAG, "Hemos encontrado " + matches.size() + " entre la template y la parte baja");
                            LinkedList<DMatch> good_matches = new LinkedList<>();
                            MatOfDMatch gm = new MatOfDMatch();

                            double minDistance = 1000;

                            int rowCount;

                            if(desc1.rows() < matchesList.size())
                                rowCount = desc1.rows();
                            else
                                rowCount = matchesList.size();

                            for (int i = 0; i < rowCount; i++) {
                                double dist = matchesList.get(i).distance;
                                if (dist < minDistance) {
                                    minDistance = dist;
                                }
                            }

                            LinkedList<DMatch> good_matches_reduced = new LinkedList<>();
                            MatOfDMatch gmr = new MatOfDMatch();
                            double upperBound = 2 * minDistance;
                            for (int i = 0; i < rowCount; i++) {
                                if (matchesList.get(i).distance < upperBound && good_matches.size() < 500) {
                                    good_matches.addLast(matchesList.get(i));
                                    if(i < 10 && debug)
                                    {
                                        good_matches_reduced.addLast(matchesList.get(i));
                                    }
                                }
                            }
                            gm.fromList(good_matches);
                            if(debug) {
                                gmr.fromList(good_matches_reduced);
                            }
                        }
                    }


                    gray.release();
                    if(index == (trigger_size - 1)) {
                        thread_over = true;
                    }
                }
            });
        }
    }

    private void setBase64Pattern(JSONArray dataArray) {
        detection.clear();
        Log.e(TAG, "setBase64Pattern");
        for (int i = 0; i < dataArray.length(); i++) {
            try {
                Log.e(TAG, "setBase64Pattern tratando template " + i);
                detection.add(0);
                String image_base64 = dataArray.getString(i);
                String settings = "";
                if(image_base64 != null && !image_base64.isEmpty()) {
                    Log.e(TAG, "la template no esta vacia");
                    Mat image_pattern = new Mat();
                    MatOfKeyPoint kp1 = new MatOfKeyPoint();
                    Mat desc1 = new Mat();
                    int limit = 800;
                    if(image_base64.contains("data:"))
                        image_base64 = image_base64.split(",")[1];
                    byte[] decodedString = Base64.decode(image_base64, Base64.DEFAULT);
                    Log.e(TAG, "creando bitmap");
                    Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
                    Bitmap scaled = bitmap;
                    if (bitmap.getWidth() > limit) {
                        double scale = bitmap.getWidth() / limit;
                        scaled = Bitmap.createScaledBitmap(bitmap, (int) (bitmap.getWidth()/scale), (int) (bitmap.getHeight()/scale), true);
                        if (bitmap.getHeight() > limit) {
                            scale = bitmap.getHeight() / limit;
                            scaled = Bitmap.createScaledBitmap(bitmap, (int) (bitmap.getWidth()/scale), (int) (bitmap.getHeight()/scale), true);
                        }
                    }
                    Log.e(TAG, "creando Mat");
                    Utils.bitmapToMat(scaled, image_pattern);
                    Imgproc.cvtColor(image_pattern, image_pattern, Imgproc.COLOR_BGR2GRAY);
                    //Imgproc.equalizeHist(image_pattern, image_pattern);

                    /*
                    // Change SURF configuration
                    Log.e(TAG, "Preparando archivo de configuracion de SURF templates");
                    String extStorageDirectory = Environment.getExternalStorageDirectory().toString();

                    try{
                        File tempFile = File.createTempFile("config" + i, ".yml", new File(extStorageDirectory));
                         if (i == 1){
                            settings = "%YAML:1.0\nhessianThreshold: 10000\n";
                        }else{
                            settings = "%YAML:1.0\nhessianThreshold: 1000\n";
                        }

                        FileWriter writer = new FileWriter(tempFile, false);
                        writer.write(settings);
                        writer.close();
                        Log.e(TAG, "cargando...");
                        orbDetector.read(tempFile.getPath());
                    }catch(IOException e){
                        Log.e(TAG, "error cargando la configuracion de SURF");
                        e.printStackTrace();
                    }
                    */
                    Log.e(TAG, "detectando...");
                    orbDetector.detect(image_pattern, kp1);
                    orbDescriptor.compute(image_pattern, kp1, desc1);

                    Log.e(TAG, "Tratando la template " + i + " tiene " + kp1.size() + " puntos de interes.");

                    triggers.add(image_pattern);
                    triggers_kps.add(kp1);
                    triggers_descs.add(desc1);
                }
            } catch (JSONException e) {
                Log.e(TAG, "setBase64Pattern algo ha fallado");
                // do nothing
            }
        }
    }

    private void updateState(boolean state, int _index) {
        final int index = _index;
        Log.e(TAG, "updateState");
        int detection_limit = 6;

        if(state) {
            try {
                int result = detection.get(_index) + 1;
                if(result < detection_limit) {
                    detection.set(_index, result);
                }
            } catch (IndexOutOfBoundsException ibe){
//                 detection.add(_index, 1);
            }
        } else {
            for (int i = 0; i < triggers.size(); i++) {
                try {
                    int result = detection.get(i) - 1;
                    if(result < 0) {
                        detection.set(_index, 0);
                    } else {
                        detection.set(_index, result);
                    }
                } catch (IndexOutOfBoundsException ibe){
//                    detection.add(i, 0);
                }
            }
        }

        if (getState(_index) && called_failed_detection && !called_success_detection) {
            cordova.getThreadPool().execute(new Runnable() {
                public void run() {
                    PluginResult result = new PluginResult(PluginResult.Status.OK, "{\"message\":\"pattern detected\", \"index\":" + index + "}");
                    result.setKeepCallback(true);
                    cb.sendPluginResult(result);
                }
            });
            called_success_detection = true;
            called_failed_detection = false;
            detected_index = _index;
        }

        boolean valid_index = detected_index == _index;

        if (!getState(_index) && !called_failed_detection && called_success_detection && valid_index) {
            cordova.getThreadPool().execute(new Runnable() {
                public void run() {
                    PluginResult result = new PluginResult(PluginResult.Status.ERROR, "{\"message\":\"pattern not detected\"}");
                    result.setKeepCallback(true);
                    cb.sendPluginResult(result);
                }
            });
            called_success_detection = false;
            called_failed_detection = true;
        }
    }

    private String loadAssetTextAsString(Context context, String name) {
        BufferedReader in = null;
        try {

            StringBuilder buf = new StringBuilder();
            InputStream is = context.getAssets().open(name);
            in = new BufferedReader(new InputStreamReader(is));

            String str;
            boolean isFirst = true;
            while ( (str = in.readLine()) != null ) {
                if (isFirst)
                    isFirst = false;
                else
                    buf.append('\n');
                buf.append(str);
            }
            return buf.toString();
        } catch (IOException e) {
            Log.e(TAG, "Error opening asset " + name);
            Log.e(TAG, e.getMessage());
        } finally {
            if (in != null) {
                try {
                    in.close();
                } catch (IOException e) {
                    Log.e(TAG, "Error closing asset " + name);
                }
            }
        }

        return null;
    }

    private boolean getState(int index) {
        int total;
        int detection_thresh = 3;

        total = detection.get(index);

        if(debug) {
            Log.i("## GET STATE RESULT ##", " state -> " + total);
        }

        return total >= detection_thresh;
    }
}

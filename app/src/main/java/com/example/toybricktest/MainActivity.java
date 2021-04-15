package com.example.toybricktest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import java.nio.ByteBuffer;

import xindasdk.XDSDK;

public class MainActivity extends AppCompatActivity {
    private int numClasses = 3;
    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String paramPath = Environment.getExternalStorageDirectory().getPath() + "/pl/mobilenet_eye.rknn";
        XDSDK xdsdk = new XDSDK(224, 3, numClasses, paramPath);
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.raw.eye0, options);
        byte[] bytes = Bitmap2Bytes(bitmap);
        long time1 = System.currentTimeMillis();
//        float[] results = new float[3];
//        xdsdk.native_run(bytes, results);
        float[] result = xdsdk.geteye(bytes, 224, 224, 3, 0);
        long time2 = System.currentTimeMillis();
        Log.i(TAG, "time cost = " + (time2-time1) + "\n  result = " + result[0] + " * " + result[1] + " * " + result[2]);
        xdsdk.deinit();
        xdsdk = null;
    }

    public byte[] Bitmap2Bytes(Bitmap bm) {
        ByteBuffer argb_buf = ByteBuffer.allocate(bm.getByteCount());
        bm.copyPixelsToBuffer(argb_buf);
        byte[] pixels = argb_buf.array();
        return pixels;
    }
}

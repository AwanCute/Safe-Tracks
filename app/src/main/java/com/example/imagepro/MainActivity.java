package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // This sets the view for the activity from a layout resource defined in R.layout.activity_main.
        setContentView(R.layout.activity_main);

        // This creates a new Handler to post a delayed task on the main thread.
        new Handler().postDelayed(new Runnable() {
            // This method will be called after the specified delay.
            @Override
            public void run() {
                // This intent is used to start CameraActivity. It clears the current task and creates a new task.
                startActivity(new Intent(MainActivity.this, CameraActivity.class)
                        .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP));
                // This call finishes the current Activity so it's removed from the back stack.
                finish();
            }
            // The Runnable will be executed after a delay of 5000 milliseconds (5 seconds).
        }, 5000);
    }
}

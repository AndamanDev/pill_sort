package com.pharmaflow.pillcount

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.util.Log

/**
 * Kill the counting app and start it again, from a process that is not the one dying.
 *
 * WHY A SECOND PROCESS AT ALL. What has to be thrown away is not the Activity but the
 * PROCESS: android.hardware.camera2.CameraManager keeps its list of cameras per process,
 * and after a USB camera is unplugged and plugged back in that list still names a device
 * that is gone -- "Unknown camera ID 111" is what the board logged while the camera
 * service was offering 112. Restarting the Activity in place changes nothing, because the
 * same CameraManager comes back with it. Only a new process gets a new one.
 *
 * AND WHY IT CANNOT BE DONE FROM THE APP ITSELF. The first attempt set an AlarmManager to
 * re-launch and then exited. Android 10 and up refuse an Activity start that comes from
 * the background, the alarm's start was refused, and the app simply vanished off the bench
 * with no message and nothing to press. This Activity is in the FOREGROUND when it starts
 * the counter again, so the start is allowed; it runs in `:restart` (see the manifest) so
 * killing the main process does not kill the thing doing the killing.
 *
 * It draws nothing and lives for a fraction of a second.
 *
 * TRANSLUCENT, NOT Theme.NoDisplay, and the difference is the whole activity working or
 * not. NoDisplay requires finish() to have been called BEFORE onResume() returns; this one
 * has to outlive onResume by the beat it takes the old process to die, so Android threw
 *
 *     IllegalStateException: Activity RestartActivity did not call finish() prior to
 *     onResume() completing
 *
 * and killed this process -- after it had already killed the counter and before it could
 * start it again. The counter went off the bench and did not come back, which is the one
 * outcome worse than the camera fault this is here to fix.
 */
class RestartActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val victim = intent.getIntExtra(EXTRA_PID, 0)
        Log.w(TAG, "restarting the counter; killing pid $victim")
        if (victim > 0 && victim != Process.myPid()) {
            Process.killProcess(victim)             // same uid, so this is permitted
        }
        // A beat for the process to actually go. Launching into a process that has not
        // finished dying is how you get the old one handed back, CameraManager and all,
        // which would make the whole exercise pointless.
        Handler(Looper.getMainLooper()).postDelayed({
            val back = Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            startActivity(back)
            finish()
        }, 600)
    }

    companion object {
        const val EXTRA_PID = "kill-pid"
        private const val TAG = "pillsort"
    }
}

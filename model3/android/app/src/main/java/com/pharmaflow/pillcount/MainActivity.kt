package com.pharmaflow.pillcount

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.camera2.CameraManager
import android.hardware.display.DisplayManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.Gravity
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.Executors

/**
 * DrugCount's bench window, on a phone.
 *
 * What this class does NOT do is worth stating first, because it is most of the design:
 * it does not draw the count, the target, the buttons, the tray outlines or the status
 * pill. `ui/screen.py` composes all of that, exactly as it does on a PC, and this
 * activity's display is that image in an ImageView. A tap is mapped back into canvas
 * coordinates and handed to `ui/screen.py`'s own hit-testing, so the buttons are
 * wherever it drew them.
 *
 * The camera is the one exception, and it was not always. `compose()` used to paste the
 * video into that same image, which meant the preview could only refresh once the model
 * had finished with the frame: 1.3 fps on the test handset, of which 374 ms was the
 * forward pass. No amount of tuning fixes that -- four times faster is still 5 fps --
 * because the picture was chained to the pipeline at all. So the video is a CameraX
 * [PreviewView] now, composited by the display underneath a `compose(overlay = true)`
 * canvas that leaves the pane transparent. The camera runs at the display's rate. The
 * dots and the count arrive a frame or two behind it, which is what they are.
 *
 * The canvas is 1440x720 and landscape, which is the shape the bench window has and the
 * shape a tray is. Its size and the video pane inside it both come back from `start()`
 * rather than being written down on this side as well.
 *
 * The loop per frame:
 *
 *     CameraX (RGBA_8888)  ->  pillcount_android.frame()  ->  RGBA bytes  ->  Bitmap
 *
 * with the model's forward pass alone stepping back out to Kotlin, into [OrtEngine].
 * That analysis stream is [ANALYSIS] rather than anything larger: it is no longer what
 * anyone looks at, only what the detector reads, and it is letterboxed to 480x640 the
 * moment it reaches Python.
 */
class MainActivity : ComponentActivity() {

    private lateinit var canvasView: ImageView
    private lateinit var statusView: TextView
    private lateinit var previewView: PreviewView

    /** The splash: see [showLoading]. Every part of it is held, because every part moves. */
    private lateinit var loadingView: View
    private lateinit var loadingLogo: ImageView
    private lateinit var loadingTitle: TextView
    private lateinit var loadingBar: ProgressBar
    private lateinit var loadingPct: TextView
    private lateinit var loadingNote: TextView

    /** Set the first time a real frame reaches the screen, so the splash leaves once. */
    private var loadingDone = false

    /**
     * When the last camera frame ARRIVED, on the monotonic clock. Read by [checkCamera].
     *
     * Stamped on the way in rather than on the way out: a forward pass is most of a second
     * on this handset, and a stall measured from the end of the work would be declared on
     * every slow frame -- the app would announce a broken camera because its own model is
     * thinking. [analysing] covers the same hole from the other side.
     */
    @Volatile private var lastFrameAt = 0L

    /** True while a frame is inside the pipeline, so its own cost is not read as a stall. */
    @Volatile private var analysing = false

    /** When the camera was last rebound, so a dead camera is not rebound every tick. */
    private var lastRebindAt = 0L

    /**
     * When [startCamera] last ASKED for the camera, whether or not it got one.
     *
     * THE CLOCK [checkCamera] USES BEFORE THERE IS A FRAME TO USE. [lastFrameAt] is
     * stamped by the analyzer, so it is still 0 when the very first bind failed -- and on
     * a board with a USB camera that is not a rare case, it is what happens every time the
     * app is opened before the lead is pushed in. bindToLifecycle throws
     * "Provided camera selector unable to resolve a camera for the given use case", the
     * screen says so, and the watchdog used to read the 0 as "not started yet, nothing to
     * do" and never look again. Plugging the camera in did nothing; only restarting the
     * app helped. This gives the watchdog something to measure in that state.
     */
    @Volatile private var cameraAskedAt = 0L

    /**
     * How many times the camera has been asked for since a frame last arrived.
     *
     * IT DECIDES HOW HARD TO ASK. A session the driver dropped comes back from a plain
     * rebind, which is cheap. A camera that was UNPLUGGED does not, and the board's log
     * says so in a way nothing else would have: after the lead came out and went back in,
     * every rebind reported "cameras available: 1" and bound without throwing, and not one
     * frame arrived. That 1 was the camera CameraX enumerated at launch -- a handle onto a
     * device that no longer exists -- because the provider never looks at the hardware
     * twice. So the first try is a rebind and every try after it is a rescan.
     */
    @Volatile private var rebinds = 0

    /** True between asking CameraX to shut down and giving up or getting an answer. */
    @Volatile private var rescanning = false

    /** Said once per outage, so the message is not rewritten every six seconds. */
    @Volatile private var cameraLostSaid = false


    /** The display rotation the camera is currently aimed for. -1 until it is bound. */
    private var boundRotation = -1

    /** Kept so a turn of the phone can re-aim them without rebuilding the pipeline. */
    private var analysisUseCase: ImageAnalysis? = null
    private var previewUseCase: Preview? = null

    /** Is the picture being shown left-to-right reversed. Owned by Python; see applyMirror. */
    private var mirrored = false

    /** Is it shown turned half round. Owned by Python, applied with the mirror. */
    private var rotated = false

    /** How close the slider asks for, 1.0 upwards. Owned by Python; see [applyZoom]. */
    private var wantZoom = 1f

    /**
     * The part of [wantZoom] the camera could NOT do itself, and this app does instead:
     * Python crops the middle 1/softZoom of every frame, and the preview surface is scaled
     * by the same factor about the pane's centre. 1.0 whenever CameraX zoomed the lens.
     * Read on the analyzer thread, written on the UI thread.
     */
    @Volatile private var softZoom = 1f

    /**
     * Is the bound camera one CameraX cannot zoom -- in practice a USB camera on the board,
     * whatever way it claims to face. Decides two things at bind time: the analysis stream is asked for at [ANALYSIS_SOFT] so a software crop
     * has real pixels to spend, and the preview runs on a TextureView, whose scale is
     * honoured by the compositor wherever the view goes.
     */
    private var externalCamera = false

    /**
     * TURNING THE PHONE END FOR END DOES NOT CHANGE ITS CONFIGURATION, and that is the
     * whole reason this listener exists.
     *
     * This activity is sensorLandscape, so both ways round are "landscape": flipping it
     * fires no onConfigurationChanged, recreates nothing, and rebinds nothing. What does
     * change is the display's rotation, from 90 to 270 -- and ImageAnalysis keeps the
     * target rotation it was built with, so the buffer it hands Python arrives UPSIDE
     * DOWN with respect to the preview the operator is looking at.
     *
     * That is not a cosmetic fault. The preview is CameraX's own surface and is always
     * the right way up; the dots are measured on the analysis image and drawn over that
     * preview, so every dot lands 180 degrees away from its pill. Measured on the test
     * handset at ROTATION_270: a pill the model placed at (0.675, 0.323) of the frame
     * appeared on screen at (0.327, 0.675), which is that point rotated by half a turn.
     */
    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) {
            val now = previewView.display?.rotation ?: return
            if (now == boundRotation) return
            Log.i(TAG, "display rotation $boundRotation -> $now, re-aiming the analyser")
            boundRotation = now
            // RE-AIMED, NOT REBOUND. Setting targetRotation is what CameraX asks for here:
            // it changes the rotationDegrees reported on the frames that follow and costs
            // nothing. Unbinding and binding again would do the same job and take the
            // camera down for a few hundred milliseconds to do it -- a black pane and a
            // dropped count every time somebody turns the phone round on the bench.
            analysisUseCase?.targetRotation = now
            previewUseCase?.targetRotation = now
        }
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * THE CAMERA CAN STOP AND NOTHING WOULD HAVE NOTICED, which is why this exists.
     *
     * The whole screen is composed by the analyzer, so when frames stop arriving the app
     * stops running: the last picture stays up, the count beside it stays up, and both
     * look current. The bench has a repaint timer of its own and catches this in
     * app/window.py; here, this is that timer. See pillcount_android.idle().
     */
    private val watchdog = object : Runnable {
        override fun run() {
            checkCamera()
            mainHandler.postDelayed(this, WATCH_MS)
        }
    }

    private val worker = Executors.newSingleThreadExecutor()

    /**
     * Touches run here, NOT on [worker].
     *
     * They shared a thread, and a thread that is busy with a forward pass is a thread
     * that is not listening: every press waited for whatever frame was in flight, which
     * on this handset is a few hundred milliseconds, and the whole app felt like it was
     * thinking about it. A press changes a number in Python and that is all it does --
     * it has no reason to queue behind the model.
     */
    private val touchWorker = Executors.newSingleThreadExecutor()

    private var bridge: PyObject? = null
    private var engine: OrtEngine? = null

    /** The export whose breadcrumb is still on disk, until a real frame rubs it out. */
    @Volatile
    private var modelPending: String? = null

    private var canvasW = 0
    private var canvasH = 0

    /** The video pane in canvas coordinates, straight from `ui.screen.pane()`. */
    private var pane = intArrayOf(0, 0, 0, 0, 0)

    /**
     * Two bitmaps, swapped. The worker thread fills one while the UI thread is drawing
     * the other; with a single bitmap a 4 MB pixel copy lands underneath a draw in
     * progress and the screen tears across a band of the image.
     */
    private var bitmaps: Array<Bitmap>? = null
    private var next = 0

    /** Reused between frames. A fresh 3.7 MB array per frame is pure GC pressure. */
    private var frameBuf = ByteArray(0)

    /** The geometry line is worth one entry in the log and not one per frame. */
    private var loggedGeometry = false

    /**
     * The bound camera, kept for its [androidx.camera.core.CameraControl].
     *
     * The bench window drives a UVC webcam's SHUTTER TIME through OpenCV; there is no
     * equivalent here. CameraX offers exposure COMPENSATION -- an index in device-defined
     * EV steps over a range the device reports -- which is a different control with a
     * different meaning, and that is exactly why `ui/screen.py` takes the slider's label
     * as text instead of working it out for itself.
     */
    private var camera: Camera? = null

    /**
     * Long press is this app's substitute for the bench's `r` key: a phone has no
     * keyboard, and the counting frame has to be started and saved somehow. A tap on the
     * video would do it by accident; a long press is the standard "edit this" gesture and
     * cannot be hit while reaching for a button.
     */
    private var gestures: GestureDetector? = null

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else status("ไม่ได้รับสิทธิ์กล้อง\nCamera permission denied")
    }

    /**
     * MEASUREMENT ONLY. It changes nothing; it writes down what the system says.
     *
     * The question it answers: when a USB camera is unplugged and plugged back in, can
     * THIS PROCESS see the new one at all? The board's log says the app ends up asking
     * cameraserver about camera "110" after a replug and being told there is no such
     * camera -- but that ID came out of a list the process already had, so it proves the
     * list is stale without proving the process is blind. If these callbacks report the
     * new id, the process can see it and the fix is to make CameraX look again. If they
     * report nothing, nothing in this process will ever see that camera and only a new
     * process can.
     *
     * Registered for the life of the activity so both edges are caught: the unplug and
     * the plug-in. It is a listener on the framework's own camera service events, not a
     * poll of the cached list -- which is the whole point of asking it.
     */
    private val cameraWatch = object : CameraManager.AvailabilityCallback() {
        override fun onCameraAvailable(id: String) {
            Log.w(TAG, "system says camera available: $id   ids now=${idList()}")
            if (lastFrameAt != 0L && SystemClock.elapsedRealtime() - lastFrameAt < STALL_MS) {
                return                              // already counting; nothing to do
            }
            // A CAMERA HAS APPEARED AND CAMERAX DOES NOT KNOW. Its list was built at the
            // last getInstance() and nothing makes it look again, so this is the moment to
            // make it -- and the only moment there is, because the periodic retry gives up
            // after GIVE_UP_TRIES and would otherwise never ask once more.
            rebinds = 0
            cameraLostSaid = false
            rescanCameras()
        }
        override fun onCameraUnavailable(id: String) {
            Log.w(TAG, "system says camera unavailable: $id   ids now=${idList()}")
        }
    }

    /** What the framework's list holds right now, for comparing against CameraX's view. */
    private fun idList(): String = try {
        getSystemService(CameraManager::class.java).cameraIdList.joinToString(",")
    } catch (t: Throwable) {
        "<${t.javaClass.simpleName}>"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        getSystemService(CameraManager::class.java)
            .registerAvailabilityCallback(cameraWatch, mainHandler)
        Log.w(TAG, "camera ids at startup: ${idList()}")
        goFullscreen()
        canvasView = findViewById(R.id.canvas)
        statusView = findViewById(R.id.status)
        previewView = findViewById(R.id.preview)
        loadingView = findViewById(R.id.loading)
        loadingLogo = findViewById(R.id.loadingLogo)
        loadingTitle = findViewById(R.id.loadingTitle)
        loadingBar = findViewById(R.id.loadingBar)
        loadingPct = findViewById(R.id.loadingPct)
        loadingNote = findViewById(R.id.loadingNote)

        // FILL_CENTER is `ui.draw.cover`'s rule -- scale to fill, centre-crop the
        // overflow -- so the surface frames the scene the same way the pasted video
        // used to, and `screen.cover_map` can place the dots on it.
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // The overlay is what is touched; the surface below it must never take the tap.
        //
        // Every phase goes to Python, not just the release. The counting frame is dragged
        // and the brightness slider is dragged, and a drag reconstructed from its last
        // point is not a drag. The STATE MACHINE is on the Python side -- which corner is
        // held, where the rectangle is -- so the bench and the phone run one copy of it
        // and cannot disagree about what a drag means.
        gestures = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onLongPress(e: MotionEvent) {
                touch("long", canvasView, e)
            }
        })
        canvasView.setOnTouchListener { view, event ->
            gestures?.onTouchEvent(event)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> touch("down", view, event)
                MotionEvent.ACTION_MOVE -> touch("move", view, event)
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                    touch("up", view, event)
            }
            true
        }

        // The pane is in canvas coordinates, and the canvas is letterboxed into the
        // ImageView -- so where it lands on screen is only known once the view has been
        // measured, and again after any resize.
        canvasView.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> placePreview() }

        showLoading()
        worker.execute { boot() }
    }

    /**
     * Everything that must happen before the first frame, on the worker thread.
     *
     * Chaquopy's first start on a new install extracts CPython's standard library,
     * numpy and OpenCV out of the APK -- seconds, not milliseconds -- and building the
     * ONNX Runtime session is another second on top. Doing it here rather than lazily
     * in the analyzer means the camera does not open onto a frozen screen.
     */
    /** The name both the provider race and the crash ban file this export under. */
    private fun modelKeyOf(bytes: ByteArray): String {
        val crc = java.util.zip.CRC32().apply { update(bytes) }.value
        return "$MODEL_KEY_SALT-$crc-${bytes.size}"
    }

    private fun boot() {
        try {
            // The version AND when this copy was installed. Two builds of 0.1.0 are the
            // same version and not the same assets, and the phone has to notice.
            val pkg = packageManager.getPackageInfo(packageName, 0)
            @Suppress("DEPRECATION")
            val version = "${pkg.versionCode}-${pkg.lastUpdateTime}"
            val artDir = Art.unpack(this, version)
            step(20, "เตรียมไฟล์หน้าจอ")

            val prefs = getSharedPreferences(ENGINE_PREFS, MODE_PRIVATE)
            var modelBytes = assets.open(MODEL_FAST).use { it.readBytes() }
            var modelKey = modelKeyOf(modelBytes)

            // A BREADCRUMB FOR THE EXPORT, for the reason OrtEngine keeps one for the
            // provider: what this guards against kills the process, so it cannot be
            // caught, only found afterwards. Shipped as int8 once before, this app closed
            // the moment the loading screen ended and left nothing behind at all -- no
            // dialog, no log line, no way to tell a bad graph from a bad driver.
            //
            // KEYED BY THE BYTES, not by the file name. A rebuilt int8 export is a
            // different graph and has earned its own chance; the one that actually died
            // stays dead. It is the same key the provider race files its answer under, so
            // both memories move together when the export changes.
            val pending = prefs.getString(KEY_MODEL_PENDING, null)
            if (pending != null) {
                Log.w(TAG, "the export $pending did not survive the last run")
                prefs.edit().remove(KEY_MODEL_PENDING)
                    .putBoolean("$KEY_MODEL_BAD-$pending", true).apply()
            }
            if (prefs.getBoolean("$KEY_MODEL_BAD-$modelKey", false)) {
                modelBytes = assets.open(MODEL_SAFE).use { it.readBytes() }
                modelKey = modelKeyOf(modelBytes)
                Log.i(TAG, "float32 export: the quick one crashed this board before")
            } else {
                prefs.edit().putString(KEY_MODEL_PENDING, modelKey).apply()
                modelPending = modelKey
            }
            step(30, "วัดความเร็วโมเดล")

            // THE RACE IS REMEMBERED AGAINST THE MODEL, NOT AGAINST THE INSTALL.
            //
            // It used to share the assets' stamp -- versionCode plus lastUpdateTime -- on
            // the reasoning that a new APK brings a new model. True of some APKs and false
            // of most: lastUpdateTime changes every single time anything is sideloaded, so
            // a rebuild that touched a button's colour threw the answer away and raced
            // again. That race builds three ONNX sessions and puts nine forward passes
            // through them, and on this bench a pass is 556 ms -- five seconds of loading
            // screen, plus whatever NNAPI spends compiling the graph for the device,
            // charged to every install during a week of them.
            //
            // What the answer actually depends on is the bytes of the graph. CRC32 of
            // them, and their length, is what the key is now made of: change the export
            // and the race runs, change a caption and it does not.
            //
            // The one thing this does NOT notice is ONNX Runtime itself being upgraded,
            // which is a line in build.gradle and a thing somebody does on purpose. Bump
            // MODEL_KEY_SALT when that happens.
            Log.i(TAG, "model key $modelKey")
            val ort = OrtEngine(modelBytes, prefs, modelKey)
            engine = ort
            step(45, "โหลดโมเดลตรวจจับเม็ดยา")

            if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
            val module = Python.getInstance().getModule("pillcount_android")
            step(70, "โหลดไลบรารีประมวลผลภาพ")
            val info = JSONObject(
                module.callAttr("start", artDir.absolutePath,
                    File(getExternalFilesDir(null), "records").absolutePath,
                    ort,
                    // THE VIEW'S size, not the display's. displayMetrics reports the
                    // screen, which is not the same rectangle: on a phone with a cutout
                    // or with the gesture bar still laid out, it is a few dozen pixels
                    // taller than the window the canvas is actually stretched into, and
                    // a canvas of the wrong shape is letterboxed with exactly the grey
                    // bands this was meant to remove. The view is measured by now --
                    // boot() runs after the first layout -- and falls back to the
                    // display only if it somehow is not.
                    canvasView.width.takeIf { it > 0 }
                        ?: resources.displayMetrics.widthPixels,
                    canvasView.height.takeIf { it > 0 }
                        ?: resources.displayMetrics.heightPixels).toString()
            )
            canvasW = info.getInt("width")
            canvasH = info.getInt("height")
            val paneJson = info.getJSONArray("pane")
            pane = IntArray(paneJson.length()) { paneJson.getInt(it) }
            bitmaps = Array(2) { Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888) }
            bridge = module
            mirrored = info.optBoolean("flip", false)
            rotated = info.optBoolean("rotate", false)
            wantZoom = info.optDouble("zoom", 1.0).toFloat()
            step(88, "เตรียมการนับ")

            Log.i(TAG, "ready: $info")
            runOnUiThread {
                step(95, "เชื่อมต่อกล้อง")
                mainHandler.postDelayed(watchdog, WATCH_MS)
                applyMirror()
                placePreview()
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
                ) startCamera() else askCamera.launch(Manifest.permission.CAMERA)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "startup failed", t)
            status("เริ่มระบบไม่สำเร็จ\n${t.javaClass.simpleName}: ${t.message}")
        }
    }

    private fun startCamera() {
        cameraAskedAt = SystemClock.elapsedRealtime()
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                // THE ROTATION THE SCREEN IS AT RIGHT NOW, asked once and given to both
                // use cases. Left unset, ImageAnalysis keeps whatever the display was
                // doing when it was constructed, which on a phone that has been turned
                // since is a buffer half a turn away from the picture. See
                // [displayListener].
                val rotation = previewView.display?.rotation ?: Surface.ROTATION_0

                // THE CAMERA IS CHOSEN FIRST NOW, because what kind it is decides how big
                // a stream to ask for. See [externalCamera].
                val provider = future.get()
                val selector = pickCamera(provider)
                if (selector == null) {
                    // Not a fault to dress up in a stack trace: there is no camera on this
                    // machine at this moment. checkCamera keeps asking, and the moment a
                    // lead is pushed in the rescan below finds it.
                    status("เปิดกล้องไม่สำเร็จ\nไม่พบกล้อง  ตรวจสายกล้อง")
                    return@addListener
                }
                // ASKED OF THE CAMERA, NOT OF WHICH WAY IT FACES. The first version went by
                // the selector -- not back, not front, so USB -- and the board's UVC camera
                // answered that it faces BACK, was taken for a handset lens, got the small
                // stream, and then reported a maxZoomRatio of 1.0 once bound: zoomed by
                // enlarging 640 pixels. What matters is whether CameraX can zoom it, and
                // the CameraInfo says so before anything is bound.
                val maxZoom = try {
                    selector.filter(provider.availableCameraInfos).firstOrNull()
                        ?.zoomState?.value?.maxZoomRatio ?: 1f
                } catch (t: Throwable) {
                    1f
                }
                externalCamera = maxZoom < ZOOM_X
                Log.i(TAG, "camera maxZoomRatio $maxZoom -> " +
                        if (externalCamera) "software zoom, ${ANALYSIS_SOFT}" else "camera zoom")
                // A TextureView for a USB camera, so the surface can be scaled for the
                // software zoom; the handset's own lens zooms itself and keeps the faster
                // default. Set before the surface provider, which is when it is read.
                previewView.implementationMode = if (externalCamera)
                    PreviewView.ImplementationMode.COMPATIBLE
                else PreviewView.ImplementationMode.PERFORMANCE

                val analysis = ImageAnalysis.Builder()
                    .setTargetRotation(rotation)
                    // RGBA rather than the YUV_420_888 default. The conversion is
                    // CameraX's, done in native code on the way out, and it saves
                    // reassembling three planes with their own strides in Python for
                    // every frame.
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    // Small, now that this stream is not also the picture on screen.
                    // The model letterboxes whatever it gets to 448x448 and the colour
                    // measurements read a few hundred pixels per pill, so 720p was
                    // being paid for in cvtColor, rotate and the BGR->LAB conversion --
                    // 56 + 40 ms a frame -- and then thrown away.
                    // Except on a USB camera, which cannot zoom itself and is zoomed by
                    // cropping: there the stream is twice the size, so a crop to the
                    // middle is still real pixels -- see ANALYSIS_SOFT. Python brings
                    // every frame back down to 640 across either way.
                    .setTargetResolution(if (externalCamera) ANALYSIS_SOFT else ANALYSIS)
                    // The pipeline is slower than the camera. Keeping the latest frame
                    // and dropping the rest is what makes the screen show now rather
                    // than a queue of the recent past.
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(worker, ::analyze)

                val preview = Preview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                // A ViewPort, and it is load-bearing rather than tidy.
                //
                // The dots are measured on the analysis stream and drawn over the
                // preview surface, so the two have to be showing the same thing. The
                // first attempt simply asked both for 16:9 and assumed that settled it.
                // It does not: setTargetResolution is best-effort, the camera is free
                // to answer a 4:3 stream, and the analyzer then measures a wider field
                // of view than the preview displays -- which puts every dot off its
                // pill by a margin that grows towards the edges of the frame.
                //
                // A ViewPort states the requirement instead of hoping for it. CameraX
                // gives every use case in the group a crop rectangle covering the same
                // region, whatever resolutions it picked, and hands it over on each
                // frame as ImageProxy.cropRect. Crop to that and the analysis image IS
                // what the surface is showing, at the pane's own aspect ratio.
                val viewPort = ViewPort.Builder(Rational(pane[2], pane[3]), rotation)
                    .setScaleType(ViewPort.FILL_CENTER)   // PreviewView's scale type
                    .build()

                analysisUseCase = analysis
                previewUseCase = preview

                val group = UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .build()

                val bound = provider.let {
                    it.unbindAll()
                    it.bindToLifecycle(this@MainActivity, selector, group)
                }
                camera = bound
                boundRotation = rotation

                // There WAS an exposure slider here, and the plumbing for it outlived the
                // control. The phone's screen was rebuilt to match the bench's, which has
                // no such slider -- the bench sets a webcam's shutter once at startup and
                // leaves it -- so `set_exposure_range` went out of the Python side and
                // this call stayed behind, throwing AttributeError into the log on every
                // single launch and telling nobody anything. Dead code that fails loudly
                // is still dead code.

                runOnUiThread {
                    statusView.visibility = View.GONE
                    placePreview()
                    // A bind starts the camera at 1x: put the slider's zoom back on it,
                    // after a rebind as much as at launch.
                    applyZoom()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "camera failed", t)
                status("เปิดกล้องไม่สำเร็จ\n${t.javaClass.simpleName}: ${t.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Which camera to bind: the back lens if there is one, otherwise whatever is there.
     *
     * DEFAULT_BACK_CAMERA ALONE WAS WRONG FOR THIS HARDWARE, and it is worth being exact
     * about why, because the symptom names neither cause. That selector asks for a lens
     * whose facing is BACK. Every phone has one. A USB camera on a board is under no
     * obligation to say it faces anywhere -- CameraX files it as EXTERNAL -- and then the
     * selector resolves nothing, bindToLifecycle throws "Provided camera selector unable
     * to resolve a camera for the given use case", and the screen says the camera could
     * not be opened while /dev/video0 sits there working perfectly. The bench has no such
     * notion: app/worker.py opens device 0 and asks it nothing about which way it points.
     *
     * The count is drawn from the picture, not from the lens, so any camera will do.
     */
    private fun pickCamera(provider: ProcessCameraProvider): CameraSelector? {
        val cameras = provider.availableCameraInfos
        Log.i(TAG, "cameras available: ${cameras.size}   system ids=${idList()}")
        if (cameras.isEmpty()) return null
        if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) {
            return CameraSelector.DEFAULT_BACK_CAMERA
        }
        if (provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
            return CameraSelector.DEFAULT_FRONT_CAMERA
        }
        Log.w(TAG, "no camera faces anywhere; taking the first one CameraX lists")
        return cameras[0].cameraSelector
    }

    /**
     * Throw CameraX away so it looks at the hardware again.
     *
     * CAMERAX ENUMERATES ONCE, at the first getInstance(), and hands back the same camera
     * list for the life of the process. Open this app with the USB lead out and that list
     * stays EMPTY however long the lead is in afterwards. It is not a guess: with the
     * camera plugged in and the system healthy, Android answered
     *
     *     dumpsys media.camera  ->  Number of camera devices: 1
     *     pickCamera            ->  cameras available: 0
     *
     * every six seconds, for minutes. The operating system had the camera; CameraX was
     * still reading an answer it wrote down at launch.
     *
     * NOTHING HERE BLOCKS, and that is not decoration. The first version of this ran
     * `getInstance().get().shutdown().get()` on [worker] -- the single thread that also
     * composes the screen. When the shutdown did not come back, the thread was held for
     * ever, every later redraw queued behind it, and the app froze at 95% with
     * "เชื่อมต่อกล้อง" showing. A blocking get on a shared executor is a deadlock waiting
     * for the right day, and that day was the same afternoon.
     *
     * The timeout is for the other half of that lesson: a shutdown that never finishes
     * must not stop the app asking again.
     */
    private fun rescanCameras() {
        if (rescanning) return                  // one at a time; they are not idempotent
        rescanning = true
        val main = ContextCompat.getMainExecutor(this)
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                // UNBIND FIRST, AND THIS IS THE DIFFERENCE BETWEEN THE TWO FAILURES.
                //
                // Opened with no camera at all, nothing is bound and shutdown() came back
                // in 3 ms. Unplugged MID-SESSION there are use cases still attached to a
                // camera that no longer exists, shutdown() has to tear that session down
                // first, and it never finished once in ten tries -- the board's log says
                // "shutdown did not finish" every six seconds. Releasing the dead session
                // is what there is to do about that.
                provider.unbindAll()
                provider.shutdown().addListener({
                    Log.w(TAG, "camera provider shut down; enumerating again")
                    finishRescan()
                }, main)
            } catch (t: Throwable) {
                Log.w(TAG, "camera provider shutdown failed", t)
                finishRescan()
            }
        }, main)
        mainHandler.postDelayed({
            if (rescanning) {
                Log.w(TAG, "camera provider shutdown did not finish; asking anyway")
                finishRescan()
            }
        }, REBIND_MS)
    }

    /**
     * Say the camera is not coming back. SAY IT -- do not act on it.
     *
     * THIS METHOD USED TO RESTART THE PROCESS, and that was the worst thing this app has
     * done. AlarmManager fired the PendingIntent, Android 10 and up refused the activity
     * start because it came from the background, and the app simply exited and stayed
     * gone: a counter that vanishes off the bench mid-shift, with no message and nothing
     * to press. Broken and visible beats gone.
     *
     * It was also built on a measurement that did not say what I read into it. A restart
     * DID recover the board once -- but that was the app opened before the lead was
     * plugged in, not a lead pulled mid-session. Those are different failures: the second
     * one leaves the board's own external-camera HAL stuck (futex_wait_queue_me one time,
     * vb2_core_dqbuf the next), and no new process of ours can mend a wedged HAL.
     *
     * So the honest thing is a sentence the operator can act on, and [banked] is asked
     * only to make that sentence true about their tablets.
     */
    private fun sayCameraLost() {
        if (cameraLostSaid) return
        cameraLostSaid = true
        val atRisk = try {
            bridge?.callAttr("banked")?.toJava(Int::class.java) ?: 0
        } catch (t: Throwable) {
            // Assume the worst: if the count cannot be asked about, it is not safe to
            // throw the process away.
            Log.w(TAG, "could not ask what is banked; assuming something is", t)
            1
        }
        Log.w(TAG, "camera did not come back after $rebinds tries; banked=$atRisk")

        if (atRisk > 0) {
            // TABLETS ALREADY IN THE BOTTLE OUTRANK THE CAMERA. A restart discards
            // `rounds`, and those pours cannot be counted again by looking at anything.
            status("กล้องไม่สามารถใช้งานได้\nเก็บไว้ $atRisk เม็ด  บันทึกผลก่อน " +
                   "แล้วปิดเปิดโปรแกรมใหม่")
            return
        }

        // NOT WHILE SOMEBODY IS READING THE RECORDS. The camera is the counting page's
        // business; a restart would throw a reader out of a list they were halfway down to
        // fix something they are not looking at.
        val showing = try {
            bridge?.callAttr("page")?.toString() ?: ""
        } catch (t: Throwable) {
            Log.w(TAG, "could not ask which page is showing", t)
            ""
        }
        if (showing != "count") {
            Log.w(TAG, "camera gone but the $showing page is up; not restarting")
            return
        }

        // ONCE PER OUTAGE, NOT ONCE EVERY FOUR SECONDS. The restart only helps when the
        // board's camera HAL is alive and has simply renumbered the device; when the HAL
        // itself is wedged -- which is what an unplug mid-stream does here, four times out
        // of four -- Android has no camera to give and a new process sees exactly the same
        // nothing. Without this guard the counter would bounce off the bench for ever.
        val prefs = getSharedPreferences(ENGINE_PREFS, MODE_PRIVATE)
        val since = System.currentTimeMillis() - prefs.getLong(KEY_RESTARTED_AT, 0L)
        if (since < RESTART_COOLDOWN_MS) {
            Log.w(TAG, "already restarted ${since / 1000}s ago; not doing it again")
            status("กล้องไม่สามารถใช้งานได้\nเปิดโปรแกรมใหม่แล้วแต่กล้องยังไม่กลับมา  " +
                   "ตรวจสายกล้อง หรือปิดเปิดเครื่อง")
            return
        }

        status("กล้องไม่สามารถใช้งานได้\nกำลังเริ่มโปรแกรมใหม่")
        prefs.edit().putLong(KEY_RESTARTED_AT, System.currentTimeMillis()).apply()
        Log.w(TAG, "restarting the process to clear the stale camera list")
        // A beat so the message is on the screen before the screen goes: the operator has
        // to know WHY the app went away, or a counter that restarts itself is just a
        // counter that crashes.
        mainHandler.postDelayed({
            startActivity(Intent(this, RestartActivity::class.java)
                              .putExtra(RestartActivity.EXTRA_PID, android.os.Process.myPid()))
        }, 1500)
    }

    private fun finishRescan() {
        if (!rescanning) return                 // the timeout and the listener race
        rescanning = false
        startCamera()
    }

    private fun analyze(image: ImageProxy) {
        val py = bridge
        val bufs = bitmaps
        if (py == null || bufs == null) {
            image.close()
            return
        }
        lastFrameAt = SystemClock.elapsedRealtime()
        rebinds = 0                     // a frame is the only proof that asking worked
        if (cameraLostSaid) {
            // The outage is over, whatever ended it. Forget the restart that was spent on
            // it so the next one is not refused by the cooldown.
            cameraLostSaid = false
            getSharedPreferences(ENGINE_PREFS, MODE_PRIVATE)
                .edit().remove(KEY_RESTARTED_AT).apply()
        }
        analysing = true
        try {
            val plane = image.planes[0]
            val src = plane.buffer
            if (frameBuf.size != src.remaining()) frameBuf = ByteArray(src.remaining())
            src.get(frameBuf)

            val crop = image.cropRect
            if (!loggedGeometry) {
                loggedGeometry = true
                Log.i(TAG, "analysis ${image.width}x${image.height}" +
                        " crop=${crop.left},${crop.top},${crop.right},${crop.bottom}" +
                        " (${crop.width()}x${crop.height()})" +
                        " rotation=${image.imageInfo.rotationDegrees}" +
                        " pane=${pane[2]}x${pane[3]}")
            }

            val rgba = py.callAttr(
                "frame", frameBuf, image.width, image.height,
                plane.rowStride, image.imageInfo.rotationDegrees,
                crop.left, crop.top, crop.right, crop.bottom,
                softZoom.toDouble()
            ).toJava(ByteArray::class.java)

            // An empty array means Python decided the screen is identical to the one
            // already on it. Skipping the upload is most of what makes the app feel
            // quick: the bitmap copy and the view invalidation are the expensive half of
            // a frame that would have shown exactly the same pixels.
            //
            // A plain if rather than an early return: this is inside `analyze`, whose
            // finally clause closes the ImageProxy, and jumping out of it past that is
            // how a camera pipeline stalls on its fourth frame.
            show(rgba)
            // A WHOLE FRAME HAS COME BACK. That is what clears OrtEngine's crash
            // breadcrumb -- not the session building, not the race, which both happen on
            // zeros before the camera is even open. Only here has the chosen provider been
            // handed real data and returned from it.
            engine?.healthy()
            // AND THE EXPORT SURVIVED WITH IT. Same frame, same proof: the graph that has
            // just returned real boxes is one this board can be trusted with.
            modelPending?.let {
                modelPending = null
                getSharedPreferences(ENGINE_PREFS, MODE_PRIVATE).edit()
                    .remove(KEY_MODEL_PENDING).apply()
                Log.i(TAG, "export $it survived a real frame")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "frame failed", t)
            status("ประมวลผลภาพไม่สำเร็จ\n${t.javaClass.simpleName}: ${t.message}")
        } finally {
            analysing = false
            image.close()
        }
    }

    /**
     * Put a composed screen up. Called from the analyzer and from the watchdog alike.
     *
     * An EMPTY array means Python decided the screen is identical to the one already on
     * it. Skipping the upload is most of what makes the app feel quick: the bitmap copy
     * and the view invalidation are the expensive half of a frame that would have shown
     * exactly the same pixels.
     *
     * Both callers run on [worker], which is a single thread -- so `next` is not shared
     * across threads and the two bitmaps cannot be swapped out from under each other.
     */
    private fun show(rgba: ByteArray) {
        val bufs = bitmaps ?: return
        if (rgba.isEmpty()) return
        val bmp = bufs[next]
        next = 1 - next
        bmp.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
        runOnUiThread {
            canvasView.setImageBitmap(bmp)
            if (!loadingDone) {
                // THE SPLASH LEAVES WHEN THERE IS SOMETHING BEHIND IT, not when boot()
                // returns. Binding the camera and waiting for its first frame is another
                // moment on top of everything boot() did, and hiding at the end of boot
                // left a bare background on screen for most of a second, which is exactly
                // the "has it crashed?" this screen exists to answer. The full bar stays
                // up for a beat so the last step is something the eye catches.
                loadingDone = true
                step(100, "พร้อมใช้งาน")
                canvasView.postDelayed({ hideLoading() }, 250)
            }
        }
    }

    /**
     * Has the camera gone quiet? Then redraw saying so, and eventually ask for it again.
     *
     * TWO SEPARATE JOBS, on two clocks. The redraw is what stops a stale count being
     * saved: Python works out the age of the picture, blocks the save button and puts a
     * red border round the video, exactly as the bench does. The rebind is the phone's
     * version of app/worker.py's `_reopen` -- unbinding and binding the use cases again
     * is what pulling the USB lead out and putting it back does by hand, and it is the
     * fix for a camera another app took, a device that was re-enumerated, or a driver
     * that dropped the session. It waits [REBIND_MS] because rebinding is not free and a
     * camera that is merely slow for a moment recovers without it.
     */
    private fun checkCamera() {
        val py = bridge ?: return
        if (analysing) return                           // mid-frame; its own cost is not a stall
        val now = SystemClock.elapsedRealtime()

        // NOTHING HAS EVER ARRIVED. Either the first bind threw -- no camera was attached
        // when the app opened -- or one was attached and has never produced a frame. There
        // is no picture to redraw and no count to protect: statusView is already showing
        // the reason. The only useful thing is to ask for the camera again, because the
        // answer changes the moment somebody pushes the lead in.
        if (lastFrameAt == 0L) {
            if (cameraAskedAt == 0L) return             // startCamera has not run yet
            if (now - cameraAskedAt < REBIND_MS) return
            if (now - lastRebindAt < REBIND_MS) return
            lastRebindAt = now
            rebinds++
            if (rebinds == 1) {
                Log.w(TAG, "camera has never sent a frame, asking for it again")
                startCamera()
            } else {
                if (rebinds > GIVE_UP_TRIES) {
                    sayCameraLost()
                } else {
                    Log.w(TAG, "camera has never sent a frame after $rebinds tries, rescanning")
                    rescanCameras()
                }
            }
            return
        }

        val gap = now - lastFrameAt
        if (gap < STALL_MS) return

        worker.execute {
            try {
                show(py.callAttr("idle").toJava(ByteArray::class.java))
            } catch (t: Throwable) {
                Log.e(TAG, "idle redraw failed", t)
            }
        }

        if (gap > REBIND_MS && now - lastRebindAt > REBIND_MS) {
            lastRebindAt = now
            rebinds++
            if (rebinds == 1) {
                // Cheap first: a dropped session, a device Windows -- or Android -- merely
                // re-enumerated, comes back from this and the camera never stops.
                Log.w(TAG, "no camera frames for $gap ms, rebinding")
                startCamera()
            } else {
                // It did not come back, so the camera in CameraX's list is not the camera
                // on the end of the lead. See [rebinds].
                if (rebinds > GIVE_UP_TRIES) {
                    sayCameraLost()
                } else {
                    Log.w(TAG, "no camera frames for $gap ms after $rebinds tries, rescanning")
                    rescanCameras()
                }
            }
        }
    }

    /**
     * Put the camera surface exactly where the overlay leaves a hole.
     *
     * `screen.pane()` reports the video pane in canvas coordinates; the canvas is drawn
     * fitCenter inside [canvasView], so the same scale and letterbox offsets the tap
     * handler undoes are applied forwards here. Both read one rectangle from Python,
     * which is what stops the hole and the surface from drifting apart when the panel
     * is resized.
     *
     * The corner radius is not applied to the surface. It does not need to be: the
     * overlay stays opaque over the corners -- `compose` punches a ROUNDED hole -- so
     * the card's corners are painted on top of the video, exactly as `rounded_paste`
     * used to repaint them.
     */
    private fun placePreview() {
        if (canvasW == 0 || pane[2] == 0) return
        val vw = canvasView.width
        val vh = canvasView.height
        if (vw == 0 || vh == 0) return

        val scale = minOf(vw.toFloat() / canvasW, vh.toFloat() / canvasH)
        val offX = (vw - canvasW * scale) / 2f
        val offY = (vh - canvasH * scale) / 2f

        val lp = previewView.layoutParams as FrameLayout.LayoutParams
        lp.width = (pane[2] * scale).toInt()
        lp.height = (pane[3] * scale).toInt()
        lp.leftMargin = (offX + pane[0] * scale).toInt()
        lp.topMargin = (offY + pane[1] * scale).toInt()
        lp.gravity = Gravity.TOP or Gravity.START
        previewView.layoutParams = lp
        previewView.visibility = View.VISIBLE
    }

    /**
     * One touch, in canvas coordinates, handed to Python.
     *
     * Undoing the letterbox here -- rather than asking the Python side to lay itself out
     * for this screen -- is what keeps one set of hit boxes: `hitboxes()` measures from
     * the same numbers `_chrome()` drew with, and neither knows a phone exists.
     *
     * The canvas is drawn fitCenter inside [canvasView], so the scale and letterbox this
     * undoes are the same ones [placePreview] applies forwards -- which is what keeps a
     * finger, a button and the video surface agreeing about where things are.
     *
     * Runs on the worker, never the UI thread: `touch` can write the counting frame to
     * disk, and a file write on the main thread is a frame dropped at best.
     */
    private fun touch(phase: String, view: View, event: MotionEvent) {
        val py = bridge ?: return
        if (canvasW == 0) return
        val scale = minOf(view.width.toFloat() / canvasW, view.height.toFloat() / canvasH)
        val x = (event.x - (view.width - canvasW * scale) / 2f) / scale
        val y = (event.y - (view.height - canvasH * scale) / 2f) / scale
        if (x < 0 || y < 0 || x > canvasW || y > canvasH) return
        touchWorker.execute {
            try {
                val reply = py.callAttr("touch", phase, x.toInt(), y.toInt()).toString()
                applyTouchReply(reply)
            } catch (t: Throwable) {
                Log.e(TAG, "touch failed", t)
            }
        }
    }

    /**
     * Act on whatever the touch changed. Python decides; this only carries it out.
     *
     * ALMOST NOTHING REACHES HERE, and that is the design: the target, the counting frame,
     * the page and the records all live on the Python side and are drawn from there on the
     * next frame, so a press changes a number over there and Kotlin has no part in it.
     *
     * THE MIRROR IS THE ONE EXCEPTION, because the picture is the one thing on this screen
     * Python does not draw. The video is CameraX's own preview surface showing through a
     * hole in the canvas -- which is what lets it run at the display's rate instead of the
     * model's -- and nothing on the Python side can turn that surface round. So the flip is
     * done in two halves: Python mirrors the frame it analyses, so the marks and the
     * counting region stay on the tablets they belong to, and this mirrors the surface, so
     * the operator sees the same picture those marks were computed from. Doing one without
     * the other puts every mark on the mirror image of its own tablet, which is a worse
     * screen than the mirrored one it set out to fix.
     */
    private fun applyTouchReply(reply: String) {
        val json = try {
            JSONObject(reply)
        } catch (t: Throwable) {
            return                                  // a reply we cannot read changes nothing
        }

        // CLOSING IS THE ACTIVITY'S ALONE. The screen is drawn by Python and every control
        // on it is answered there, but a window is Android's -- so the cross in the header
        // sets a flag, Python reports it here, and this is the line that acts on it. The
        // guard that makes it take two taps is on the Python side with the other two, where
        // it can see whether there are counted pours to warn about.
        if (json.optBoolean("quit", false)) {
            Log.i(TAG, "closing on the operator's request")
            runOnUiThread { finishAndRemoveTask() }
            return
        }

        val zoom = json.optDouble("zoom", wantZoom.toDouble()).toFloat()
        if (zoom != wantZoom) {
            wantZoom = zoom
            runOnUiThread { applyZoom() }
        }

        val want = json.optBoolean("flip", mirrored)
        val turn = json.optBoolean("rotate", rotated)
        if (want == mirrored && turn == rotated) return
        mirrored = want
        rotated = turn
        runOnUiThread { applyMirror() }
    }

    /**
     * Put the slider's zoom on the picture: the camera's own zoom as far as it goes, and
     * a crop for whatever is left.
     *
     * THE CAMERA FIRST, BECAUSE IT IS FREE AND EXACT. setZoomRatio narrows the sensor
     * crop for every use case in the group at once, so the preview and the analysed frame
     * move together and nothing on this side has to agree with anything. A handset's lens
     * reports a maxZoomRatio of several times and does the whole job.
     *
     * A USB camera on the board reports 1.0 -- CameraX files it as EXTERNAL and passes no
     * zoom through -- and the bench's own UVC camera, when it was driven directly, took a
     * zoom only before it was streaming. So the rest is done here, as on the bench: Python
     * crops the middle of each frame by [softZoom], and the surface under the hole is
     * scaled by the same factor about its centre. The overflow is under the opaque canvas,
     * which is what clips it.
     */
    private fun applyZoom() {
        val cam = camera
        val max = cam?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1f
        val hw = if (max > 1.001f) minOf(wantZoom, max) else 1f
        try {
            if (cam != null && max > 1.001f) cam.cameraControl.setZoomRatio(hw)
        } catch (t: Throwable) {
            Log.w(TAG, "camera zoom $hw refused", t)
        }
        softZoom = if (hw > 0f) wantZoom / hw else wantZoom
        Log.i(TAG, "zoom $wantZoom: camera $hw (max $max), software $softZoom")
        applyMirror()
    }

    /**
     * Mirror the preview surface, or stop mirroring it.
     *
     * scaleX on the view rather than a different CameraSelector or a transform on the
     * stream: this is a BACK camera, so nothing upstream is mirroring anything and there is
     * no setting to turn off -- the mirror is in the lens's own idea of which way round the
     * world is, and the honest fix is to draw the surface the other way round. It costs the
     * compositor nothing; the view is already being scaled to fit the hole.
     *
     * The touch handler needs no matching change: taps are taken on the overlay above this
     * surface, in canvas coordinates, and Python maps them onto the frame it has already
     * mirrored. The two meet in the frame, which is the only space they share.
     */
    private fun applyMirror() {
        // The software zoom rides on the same two scales: see [applyZoom]. The half turn
        // is both scales negated about the view's centre -- the same reflection spot()
        // makes in frame pixels -- so across flips when exactly one of the two is on.
        previewView.scaleX = if (mirrored != rotated) -softZoom else softZoom
        previewView.scaleY = if (rotated) -softZoom else softZoom
    }

    /**
     * The whole screen, with no system bars over it.
     *
     * The status bar sat across the top as an empty strip: this window has its own header
     * and no use for a second one, and on a bench the phone is a counter rather than a
     * phone. STICKY_IMMERSIVE rather than plain fullscreen so a swipe from the edge still
     * brings the bars back for a moment -- somebody has to be able to get out.
     */
    /**
     * The logo and a word, while Chaquopy unpacks CPython, numpy and OpenCV.
     *
     * That takes seconds on a new install and about one on every launch after it, and
     * until it finishes there is nothing to draw: the Python that draws this app's screen
     * is the thing being unpacked. An empty grey rectangle for a second reads as a crash,
     * so the same wordmark the header carries goes up first, from the APK's assets --
     * which Kotlin can read directly, without waiting for Art.unpack.
     */
    private fun showLoading() {
        try {
            // Read from assets, not res/: it is the same file the header draws, staged
            // out of model3/phone/assets by the build, so the wordmark on the splash and
            // the one in the app cannot come from two different pictures.
            assets.open("art/logo.png").use { input ->
                loadingLogo.setImageBitmap(android.graphics.BitmapFactory.decodeStream(input))
            }
        } catch (t: Throwable) {
            Log.w(TAG, "no logo for the loading screen", t)
            loadingLogo.visibility = View.GONE
        }
        loadingView.visibility = View.VISIBLE
        step(3, "กำลังเริ่มระบบ")
    }

    /**
     * One step of the start-up finished.
     *
     * THE PERCENTAGE IS NOT DECORATIVE, which is the note app/splash.py carries too: each
     * number is published when that step actually completes, so a bar that sits at 45%
     * says the ONNX session is the slow part, and one that sits at 20% says Chaquopy is
     * still unpacking numpy onto a new install. A spinner would say only that the phone
     * has not given up yet.
     */
    private fun step(percent: Int, what: String) = runOnUiThread {
        loadingBar.progress = percent
        loadingPct.text = "$percent%"
        loadingNote.text = what
    }

    /** Back to the real canvas: the frames that follow are the app, not a picture of it. */
    private fun hideLoading() = runOnUiThread {
        loadingView.visibility = View.GONE
        statusView.visibility = View.GONE
    }

    private fun goFullscreen() {
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_FULLSCREEN
                or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) goFullscreen()
    }

    private fun status(text: String) = runOnUiThread {
        if (loadingView.visibility == View.VISIBLE) {
            // app/splash.py's fail(), in Kotlin. The bar and the percentage mean "this is
            // still going", and once it is not they are a lie: a progress bar stopped at
            // 45% under a crash message is worse than no bar at all. The message takes the
            // step's place, in red, and the title says plainly that it did not open.
            loadingBar.visibility = View.GONE
            loadingPct.visibility = View.GONE
            loadingTitle.setTextColor(DANGER)
            loadingTitle.text = "เปิดโปรแกรมไม่สำเร็จ"
            loadingNote.setTextColor(DANGER)
            loadingNote.text = text.replace("\n", "   ")
        } else {
            statusView.text = text
            statusView.visibility = View.VISIBLE
        }
    }

    /**
     * Away: stop watching. CameraX unbinds with the lifecycle, so of course no frames are
     * arriving -- and a watchdog left running would sit there redrawing a red border onto
     * a screen nobody is looking at, once a second, for as long as the app is in the
     * background.
     */
    override fun onPause() {
        super.onPause()
        mainHandler.removeCallbacks(watchdog)
        try {
            (getSystemService(DISPLAY_SERVICE) as DisplayManager)
                .unregisterDisplayListener(displayListener)
        } catch (t: Throwable) {
            Log.w(TAG, "could not unregister the display listener", t)
        }
    }

    /**
     * The other half of the same problem: a turn that DOES change the configuration.
     *
     * Landscape to reverse landscape does not; anything through portrait does, and
     * configChanges says this activity handles it rather than being recreated. Either way
     * the answer is the same one -- bind the camera again for the rotation the screen is
     * at now -- so both routes lead here.
     */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // A turn that changes the configuration changes the display's rotation too, and
        // the listener above deals with that. What is left here is the window: the canvas
        // is re-measured and the preview surface has to be put back under its hole.
        placePreview()
    }

    /**
     * Back: watch again, from NOW.
     *
     * The clock has to be reset as well as the callback re-posted. Without it the first
     * tick reads a gap the size of however long the phone was in a pocket and announces a
     * camera fault that was never a fault, a second before the rebound camera delivers
     * its first frame and clears it again.
     */
    override fun onResume() {
        super.onResume()
        if (bridge != null) {
            lastFrameAt = SystemClock.elapsedRealtime()
            mainHandler.removeCallbacks(watchdog)
            mainHandler.postDelayed(watchdog, WATCH_MS)
        }
        (getSystemService(DISPLAY_SERVICE) as DisplayManager)
            .registerDisplayListener(displayListener, mainHandler)
    }

    override fun onDestroy() {
        try {
            getSystemService(CameraManager::class.java)
                .unregisterAvailabilityCallback(cameraWatch)
        } catch (t: Throwable) {                            // noqa: measurement only
            Log.w(TAG, "could not unregister the camera watch", t)
        }
        mainHandler.removeCallbacks(watchdog)
        touchWorker.shutdown()
        super.onDestroy()
        worker.shutdown()
        engine?.close()
    }

    private companion object {

        /** theme.py's DANGER, for the one thing on this side that has to be red. */
        val DANGER = 0xFFC0392B.toInt()

        /** Where OrtEngine remembers which execution provider won its race. */
        const val ENGINE_PREFS = "engine"

        /** Bump when ONNX Runtime is upgraded: the race's answer is about it too. */
        const val MODEL_KEY_SALT = "ort1"

        /** The quick export, opened first, and the one that has to prove itself. */
        const val MODEL_FAST = "model/pillcount-det-v3-int8.onnx"

        /** What it falls back to: the float32 graph this app has always run. */
        const val MODEL_SAFE = "model/pillcount-det-v3.onnx"

        /** Written before the first real frame, removed once one comes back. */
        const val KEY_MODEL_PENDING = "model-pending"

        /** Set when a pending export is found still on disk at the next launch. */
        const val KEY_MODEL_BAD = "model-bad"

        /** How often the watchdog looks. Cheap: a subtraction, unless something is wrong. */
        const val WATCH_MS = 500L

        /**
         * Silence this long means the picture is not live any more.
         *
         * Under app/window.py's 1.5s, deliberately: Python is the one that decides what
         * counts as stale, and it should be asked a little before its own threshold so
         * the red border appears at 1.5s rather than at 1.5s plus a tick.
         */
        const val STALL_MS = 1000L

        /** How long a dead camera is left alone before the use cases are bound again. */
        /**
         * How long a dead picture waits before the app does something about it, and how
         * long between each thing it does after that.
         *
         * FOUR SECONDS, DOWN FROM SIX, AND THE LADDER ABOVE IT IS SHORTER TOO. The old
         * ladder spent thirty seconds on five rescans before it would restart, and a day
         * of logs on this board says those rescans have never once brought a camera back:
         * what brings it back is the availability callback, which fires whenever the
         * system has a camera again and does not care which rung the retry is on. So the
         * wait was almost entirely wait -- half a minute of an operator watching a black
         * rectangle, which is a long time at a dispensing bench.
         *
         * Not shorter than this, though. A picture can stall for a second or two over a
         * USB hiccup and come back on its own, and an app that restarts itself over that
         * is worse than the stall.
         */
        const val REBIND_MS = 4000L

        /**
         * Tries that must fail before the app stops asking and starts over.
         *
         * Two, so the whole ladder is: rebind at 4s, rescan at 8s, restart at 12s. Three
         * rungs, each doing something the one before it did not, and no rung repeated for
         * the sake of looking busy.
         */
        const val GIVE_UP_TRIES = 2

        /** Where the last self-restart is remembered, so one outage buys only one. */
        const val KEY_RESTARTED_AT = "camera-restarted-at"

        /**
         * How long a self-restart counts for.
         *
         * Three minutes: longer than it takes the app to come back, look for the camera
         * and give up again, so a wedged HAL cannot turn this into a loop; short enough
         * that a second, unrelated outage later in the shift still gets its own try.
         */
        const val RESTART_COOLDOWN_MS = 180_000L
        const val TAG = "pillsort"

        /**
         * What the analyzer asks for. 16:9 to match the preview's aspect ratio -- the
         * dots are placed from this stream and drawn over that surface, and two
         * different shapes would be two different fields of view.
         */
        // 4:3, to match the pane the preview fills and the ViewPort built from it. At
        // 16:9 the ViewPort cropped a 480x360 strip out of the middle for the model to
        // read, which is both smaller than it needs and a different field of view from
        // the one the operator frames the tray in.
        val ANALYSIS = Size(640, 480)

        /**
         * The same shape, twice the size, for a camera that cannot zoom itself. At the
         * slider's 2.25x the crop is still 569 real pixels across, against the 640 the
         * frame is brought down to -- where a crop of 640x480 would hand the model a
         * 284-pixel picture blown up. The bench does the same with its UVC camera (1080p
         * cropped to the old 4:3 view, brought down to 640x480), and counted a test tray
         * exactly as the camera's own 640x480 had. NOT YET MEASURED ON THE BOARD: the cost
         * is a buffer four times larger per frame, though Python converts only the crop.
         */
        val ANALYSIS_SOFT = Size(1280, 960)

        /** screen.ZOOM_X: a camera that cannot zoom this far itself gets ANALYSIS_SOFT. */
        const val ZOOM_X = 2.25f
    }
}
